#!/usr/bin/env python
# BFE_LossFunction_with_Grid_Search.py
# Converted from Jupyter notebook
# Script for binding free energy calculations with multi-objective optimization

# Initialize all imports
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import pickle
import copy
import matplotlib.pyplot as plt
import gc
import time
import sys

from sklearn.model_selection import train_test_split

import importlib
import keras.backend as K
from tensorflow.keras import regularizers, constraints
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Read Input data
def load_data():
    # Depickle the PDB(Protein Data Bank) and read csv with data
    # Use absolute paths with FINAL-HOSTGUEST-FILES directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PDBs = pickle.load(open(os.path.join(script_dir, 'Datasets/PDBs_RDKit_BFE.pkl'), 'rb'))
    df = pd.read_csv(os.path.join(script_dir, 'Datasets/Final_data_DDG.csv'))
    return PDBs, df

# Import custom modules
def import_custom_modules():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)  # Add script directory to path
    
    import models.layers_update_mobley as layers
    from models.dcFeaturizer import atom_features as get_atom_features
    importlib.reload(layers)
    return layers, get_atom_features

# Data pre-processing functions
def extract_info(PDBs, df):
    # Iterate through each pdb and extract the information
    info = []
    for pdb in list(PDBs.keys()):
        info.append(df[df['Ids'] == pdb][['pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS', 
                                           'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
                                           'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC', 
                                           'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB', 
                                           'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF']].to_numpy()[0])
    return info

def featurize(molecule, info, atom_features_func):
    """Function takes in a molecule and information for featurization"""
    atom_features = []
    # Iterate through each atom
    for atom in molecule.GetAtoms():
        # List of features for the atom
        new_feature = atom_features_func(atom).tolist() 
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        # Store mass, atomic number, formal charge
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        # Store 3D position
        new_feature += [position.x, position.y, position.z]
        # Track neighboring atoms
        for neighbor in atom.GetNeighbors()[:2]:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [neighbor_idx]
        for i in range(2 - len(atom.GetNeighbors())):
            new_feature += [-1]

        atom_features.append(np.concatenate([new_feature, info], 0))
    return np.array(atom_features)

def prepare_data(PDBs, df, info, atom_features_func):
    # X is the featurized molecule and y is the experimental binding free energy
    X = []
    y = []
    for i, pdb in enumerate(list(PDBs.keys())):
        X.append(featurize(PDBs[pdb], info[i], atom_features_func))
        y.append(df[df['Ids'] == pdb]['Ex _G_(kcal/mol)'].to_numpy()[0])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test

# PCGrad implementation
class PCGrad(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, name="PCGrad", **kwargs):
        # Remove learning_rate from kwargs if present since new Keras doesn't accept it
        kwargs.pop('learning_rate', None)
        # Call super without learning_rate parameter
        super().__init__(name=name, **kwargs)
        self._optimizer = optimizer
        
    @property
    def learning_rate(self):
        return self._optimizer.learning_rate

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        return self._optimizer.apply_gradients(grads_and_vars, name, **kwargs)

    def build(self, var_list):
        """Build the optimizer."""
        super().build(var_list)
        if hasattr(self._optimizer, 'build'):
            self._optimizer.build(var_list)

    def get_config(self):
        config = super().get_config()
        config.update({"optimizer": tf.keras.optimizers.serialize(self._optimizer)})
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer_config = config.pop("optimizer")
        optimizer = tf.keras.optimizers.deserialize(optimizer_config, custom_objects=custom_objects)
        return cls(optimizer, **config)

    def compute_gradients(self, losses, tape, var_list, weights=None):
        """Compute PCGrad projected gradients from a list of task losses."""
        assert isinstance(losses, list), "loss must be a list of task losses"
        grads_task = []

        for loss in losses:
            grads = tape.gradient(loss, var_list)
            for g in grads:
                if g is None:
                    print("Gradient is None")
            grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, var_list)]

            grads_task.append(grads)

        # Flatten and apply projection
        def flatten(grads):
            return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

        flat_grads_task = [flatten(g) for g in grads_task]
        flat_grads_task = tf.stack(flat_grads_task)
        flat_grads_task = tf.random.shuffle(flat_grads_task)

        def project(g, others):
            for o in others:
                dot = tf.reduce_sum(g * o)
                if dot < 0:
                    g -= dot / (tf.reduce_sum(o * o) + 1e-12) * o
            return g

        projected = []
        for i in range(len(flat_grads_task)):
            others = tf.concat([flat_grads_task[:i], flat_grads_task[i+1:]], axis=0)
            projected.append(project(flat_grads_task[i], others))
        projected = tf.stack(projected)

        # Apply weights after projection if provided (Option A)
        if weights is not None:
            weighted_projected = [w * p for w, p in zip(weights, projected)]
            mean_grad = tf.reduce_sum(tf.stack(weighted_projected), axis=0)
        else:
            # Average the projected gradients
            mean_grad = tf.reduce_mean(projected, axis=0)

        # Reshape gradient back to variable shapes
        reshaped_grads = []
        idx = 0
        for v in var_list:
            shape = tf.shape(v)
            size = tf.reduce_prod(shape)
            reshaped_grads.append(tf.reshape(mean_grad[idx:idx + size], shape))
            idx += size

        # Final sanitization of gradients
        reshaped_grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in reshaped_grads]
        return list(zip(reshaped_grads, var_list))

# Define the PGGCN_Hybrid model
class PGGCN_Hybrid(tf.keras.Model):
    def __init__(self, layers_module, num_atom_features=36, r_out_channel=20, c_out_channel=128, l2=1e-4, dropout_rate=0.2, maxnorm=3.0):
        super().__init__()
        # Initialize RuleGraphConvLayer and set the out_channel to be 20 and number of features to be 36
        self.ruleGraphConvLayer = layers_module.RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        # Initialize empty list of combination rules
        self.ruleGraphConvLayer.combination_rules = []
        # Initialize the Convolutional layer. Set the out_channel to be 128 and number of features as the other out_channel (20)
        self.conv = layers_module.ConvLayer(c_out_channel, r_out_channel)
        # Set the dense layer to 32 units, relu activation, use kernel regularizer l2 and use it as a bias regularizer, and add maxnorm constraint
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', name='dense1', kernel_regularizer=regularizers.l2(l2), bias_regularizer=regularizers.l2(l2), kernel_constraint=constraints.MaxNorm(maxnorm))
        # Set the dropout rate
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        # Set dense layer to 16 units and same activation, regularizer, and maxnorm constraint
        self.dense5 = tf.keras.layers.Dense(16, activation='relu', name='dense2', kernel_regularizer=regularizers.l2(l2), bias_regularizer=regularizers.l2(l2), kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense6 = tf.keras.layers.Dense(1, name='dense6', kernel_regularizer=regularizers.l2(l2), bias_regularizer=regularizers.l2(l2), kernel_constraint=constraints.MaxNorm(maxnorm))
        # The weights have been manually set to a specific pattern based on the physics coefficients
        # Bias innitializers are set to 0 so there is no offset (rely on weights and features)
        self.dense7 = tf.keras.layers.Dense(1, name='dense7',
                                             kernel_initializer=tf.keras.initializers.Constant([.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]),
                                             bias_initializer=tf.keras.initializers.Zeros(),
                                             kernel_regularizer=regularizers.l2(l2), 
                                             bias_regularizer=regularizers.l2(l2), 
                                             kernel_constraint=constraints.MaxNorm(maxnorm))
        self.i_s = None

    # adds rule to RuleGraphConvLayer (see layers_update_mobley.py)
    def addRule(self, rule, start_index, end_index=None):
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)
    
    def set_input_shapes(self, i_s):
        self.i_s = i_s

    # It's internal and it's called everytime you call the class and sets the physics info and it's run through the labels 
    def call(self, inputs, training=True):
        physics_info = inputs[:, 0, 38:]
        #x_a = [inputs[i, :, :38] for i in range(tf.shape(inputs)[0])]

        x_a = []
        for i in range(len(self.i_s)):
            x_a.append(inputs[i][:self.i_s[i], :38])
        x = self.ruleGraphConvLayer(x_a)
        x = self.conv(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense5(x)
        x = self.dropout2(x, training=training)
        model_var = self.dense6(x)
        merged = tf.concat([model_var, physics_info], axis=1)
        out = self.dense7(merged)
        return tf.concat([out, physics_info], axis=1)

# Loss tracking variables
empirical_loss_value = tf.Variable(0.0, trainable=False, dtype=tf.float32)
physics_loss_value = tf.Variable(0.0, trainable=False, dtype=tf.float32)

# Callback class to track losses
class LossComponentsCallback_Hybrid(tf.keras.callbacks.Callback):
    def __init__(self, model_instance):
        super().__init__()
        self.empirical_losses = []
        self.physical_losses = []
        self.total_losses = []
        self.learning_rates = []
        self.model_instance = model_instance
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model_instance.optimizer.learning_rate
        self.total_losses.append(logs.get('loss'))
        self.empirical_losses.append(float(empirical_loss_value.numpy()))
        self.physical_losses.append(float(physics_loss_value.numpy()))
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model_instance.optimizer.iterations)  # Call the schedule
        else:
            lr = lr  

        self.learning_rates.append(float(tf.keras.backend.get_value(lr)))

# Loss functions
def pure_rmse_hybrid(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1])
    return K.sqrt(K.mean(K.square(y_pred - y_true_flat)))

def physical_consistency_loss(y_true, y_pred, physics_info):
    dG_pred = y_pred
    y_true = tf.reshape(y_true, (-1, 1))

    # Physical Inconsistency loss
    # Extract the components from physics_info
    host = tf.gather(physics_info, [0, 3, 6, 9, 12], axis=1)  # Host energy terms
    guest = tf.gather(physics_info, [1, 4, 7, 10, 13], axis=1)  # Guest energy terms
    complex_ = tf.gather(physics_info, [2, 5, 8, 11, 14], axis=1)  # Complex energy terms

    # Calculate ΔG based on physics: ΔG = ΔGcomplex - (ΔGhost + ΔGguest)
    dG_physics = tf.reduce_sum(complex_, axis=1, keepdims=True) - (tf.reduce_sum(host, axis=1, keepdims=True) + tf.reduce_sum(guest, axis=1, keepdims=True))
    phy_loss = K.sqrt(K.mean(K.square(dG_pred - dG_physics)))
    
    return phy_loss

# Combined loss function
def combined_loss(physics_hyperparam=0.0003):
    def loss_function(y_true, y_pred):
        # Extract prediction and physics info
        prediction = y_pred[:, 0]
        physics_info = y_pred[:, 1:16]  # Assuming 15 physical features
        
        # Calculate individual loss components
        empirical_loss = pure_rmse_hybrid(y_true, prediction)
        physics_loss = physical_consistency_loss(y_true, prediction, physics_info)
        
        # Append losses to a list and return
        losses = []
        losses.append(empirical_loss)
        losses.append(physics_loss)
        
        # Keep track each of the loss values. Total loss is returned
        # The value ones are logs for the history
        total_loss = empirical_loss + (physics_hyperparam * physics_loss)
        empirical_loss_value.assign(empirical_loss)
        physics_loss_value.assign(physics_loss) 

        return total_loss
    
    return loss_function

# Debugging utilities
def report_none_grads(grads, vars, tag):
    none_idx = [i for i,(g,_) in enumerate(zip(grads, vars)) if g is None]
    if none_idx:
        names = [vars[i].name for i in none_idx]
        print(f"[{tag}] None gradients on {len(none_idx)} vars:")
        for n in names[:10]:
            print("  -", n)
    else:
        print(f"[{tag}] No None gradients")

# Run search function - optimized version
def run_search(physics_weight, X_train, X_test, y_train, y_test, layers_module, epochs=150):
    # Aggressive session clearing
    tf.keras.backend.clear_session()
    gc.collect()
    
    start_time = time.time()
    print(f"---------- Physics Weight: {physics_weight} ------------")

    # Create fresh copies of data
    X_train_copy = copy.deepcopy(X_train)
    X_test_copy = copy.deepcopy(X_test)
    y_train_copy = copy.deepcopy(y_train)
    y_test_copy = copy.deepcopy(y_test)

    # Create NEW model instance (critical!)
    print("Creating fresh model...")
    m = PGGCN_Hybrid(layers_module)
    m.addRule("sum", 0, 32)
    m.addRule("multiply", 32, 33)
    m.addRule("distance", 33, 36)

    # Use a fresh optimizer with different random seed
    # lr_schedule = ExponentialDecay(
    #     initial_learning_rate=0.005,
    #     decay_steps=10000,
    #     decay_rate=0.9,
    #     staircase=True
    # )

    # lr_schedule = ExponentialDecay(
    #     initial_learning_rate=0.005,
    #     decay_steps=10000,
    #     decay_rate=0.9,
    #     staircase=True
    # )
    lr_schedule = 1e-4
    tf.random.set_seed(int(physics_weight * 10000))  # Different seed per weight
    opt = PCGrad(tf.keras.optimizers.Adam(learning_rate=lr_schedule))

    input_shapes = [X.shape[0] for X in X_train_copy]
    m.set_input_shapes(input_shapes)
    
    # Pad training data
    for i in range(len(X_train_copy)):
        if X_train_copy[i].shape[0] < 2000:
            X_train_copy[i] = np.concatenate([X_train_copy[i], np.zeros([2000 - X_train_copy[i].shape[0], 53])], axis=0)

    # Pad test data
    for i in range(len(X_test_copy)):
        if X_test_copy[i].shape[0] < 2000:
            X_test_copy[i] = np.concatenate([X_test_copy[i], np.zeros([2000 - X_test_copy[i].shape[0], 53])], axis=0)

    # Convert to same type
    X_train_copy = np.array(X_train_copy).astype(np.float32)
    y_train_copy = np.array(y_train_copy).astype(np.float32)
    X_test_copy = np.array(X_test_copy).astype(np.float32)
    y_test_copy = np.array(y_test_copy).astype(np.float32)

    print(f"Training data shape: {X_train_copy.shape}")
    print(f"Test data shape: {X_test_copy.shape}")

    total_losses = []
    empirical_losses = []
    physics_losses = []

    # Early stopping
    best_train_loss = float("inf")
    patience = 15
    patience_counter = 0
    min_delta = 0.001
    best_weights = None

    print("Starting training...")
    training_start = time.time()

    # Training loop with optimizations
    for ep in range(epochs):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m.trainable_variables)
            predictions = m(X_train_copy, training=True)
            
            # Calculate losses
            emp_loss = pure_rmse_hybrid(y_train_copy, predictions[:, 0])
            phy_loss = physical_consistency_loss(y_train_copy, predictions[:, 0], predictions[:, 1:16])
            total_loss = emp_loss + physics_weight * phy_loss

        # Apply gradients using Option A (weight after projection)
        weights_vec = [1.0, float(physics_weight)]
        grads_and_vars = opt.compute_gradients([emp_loss, phy_loss], tape, m.trainable_variables, weights=weights_vec)
        opt.apply_gradients(grads_and_vars)

        # Store losses
        current_total_loss = float(total_loss.numpy())
        total_losses.append(current_total_loss)
        empirical_losses.append(float(emp_loss.numpy()))
        physics_losses.append(float(phy_loss.numpy()))

        # Early stopping logic
        if current_total_loss + min_delta < best_train_loss:
            best_train_loss = current_total_loss
            best_weights = m.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {ep + 1}")
                break

        # Print progress every 50 epochs to reduce overhead
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"Epoch {ep + 1}/{epochs} - Total: {current_total_loss:.4f}, "
                  f"Empirical: {emp_loss.numpy():.4f}, Physics: {phy_loss.numpy():.4f}")
    
    training_time = time.time() - training_start
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Restore best weights
    if best_weights is not None:
        m.set_weights(best_weights)
        print("Restored best weights from training")

    # Evaluation phase
    print("Starting evaluation...")
    eval_start = time.time()
    
    # Initialize prediction list
    y_pred_test = []
    
    # Make predictions
    for i in range(len(X_test_copy)):
        try:
            m.set_input_shapes([X_test_copy[i].shape[0]])
            pred = m(X_test_copy[i][np.newaxis, ...], training=False)
            pred_value = float(pred[0, 0].numpy())
            y_pred_test.append(pred_value)
        except Exception as e:
            print(f"Warning: Prediction {i} failed: {e}")
            y_pred_test.append(0.0)  # Default value
    
    y_pred_test = np.array(y_pred_test)
    eval_time = time.time() - eval_start
    
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    print(f"Predictions shape: {y_pred_test.shape}")
    print(f"Predictions range: [{y_pred_test.min():.4f}, {y_pred_test.max():.4f}]")
    print(f"True values range: [{y_test_copy.min():.4f}, {y_test_copy.max():.4f}]")
    
    # Calculate metrics
    MAD = np.mean(np.abs((y_test_copy) - (y_pred_test)))
    # Calculate MAD without inner np.abs (mean of the signed differences)
  
    test_emp_loss = pure_rmse_hybrid(y_test_copy, y_pred_test)
    test_phy_loss = physical_consistency_loss(y_test_copy, y_pred_test, X_test_copy[:, 0, 38:53])
    test_loss = test_emp_loss + physics_weight * test_phy_loss

    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
    print(f"MAD (mean of abd differences): {MAD:.6f}")
    print(f"Test empirical loss: {float(test_emp_loss.numpy()):.6f}")
    print(f"Test physics loss: {float(test_phy_loss.numpy()):.6f}")
    print(f"Test total loss: {float(test_loss.numpy()):.6f}")

    result_temp = {
        'name': 'ΔΔG with Multi-Loss',
        'y_test': y_test_copy,
        'test_loss': float(test_loss.numpy()),
        'test_emp_loss': float(test_emp_loss.numpy()),
        'test_phy_loss': float(test_phy_loss.numpy()),
        'y_pred_test': y_pred_test,
        'MAD': MAD,
        'all_losses': total_losses,
        'empirical_losses': empirical_losses,
        'physical_losses': physics_losses,
        'training_time': training_time,
        'eval_time': eval_time,
        'total_time': elapsed,
        'hyperparameters': {
            'physics_weight': physics_weight,
            'epochs': ep + 1 if best_weights is not None else epochs,
            'initial_learning_rate': 0.005,
            'decay_steps': 10000,
            'decay_rate': 0.9
        }
    }

    # Create plots with more detail
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 3, 1)
    plt.plot(range(1, len(total_losses) + 1), total_losses, 'b-', label='Total Loss', linewidth=2)
    plt.title(f'Total Loss (Physics Weight: {physics_weight})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 2)
    plt.plot(range(1, len(empirical_losses) + 1), empirical_losses, 'r-', label='Empirical Loss', linewidth=2)
    plt.title('Empirical Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 3)
    plt.plot(range(1, len(physics_losses) + 1), physics_losses, 'g-', label='Physics Loss', linewidth=2)
    plt.title('Physics Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 4)
    plt.scatter(y_test_copy, y_pred_test, alpha=0.7, s=50)
    plt.plot([y_test_copy.min(), y_test_copy.max()], [y_test_copy.min(), y_test_copy.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs Predicted (MAD: {MAD:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 5)
    residuals = y_test_copy - y_pred_test
    plt.scatter(y_pred_test, residuals, alpha=0.7, s=50)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 6)
    plt.hist(residuals, bins=10, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 7)
    plt.scatter(y_test_copy, y_pred_test, alpha=0.7, s=50, c='purple')
    plt.plot([y_test_copy.min(), y_test_copy.max()], [y_test_copy.min(), y_test_copy.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs Predicted (MAD: {MAD:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 8)
    signed_error = y_test_copy - y_pred_test  # Not taking absolute value
    plt.hist(signed_error, bins=10, alpha=0.7, edgecolor='black', color='purple')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.axvline(x=MAD, color='g', linestyle='-', label=f'Mean: {MAD:.4f}')
    plt.xlabel('Signed Error (True - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Signed Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results_physics_weight_{physics_weight}.png')
    plt.close()

    # Explicit cleanup
    del m, X_train_copy, X_test_copy, y_train_copy
    gc.collect()
    
    return result_temp

# Full grid search function
def run_full_grid_search(X_train, X_test, y_train, y_test, layers_module, weights=[0.001, 0.0025, 0.005]):
    results = {}
    all_results = []
    best_loss = float('inf')
    best_result = None

    print("Starting optimized grid search...")
    print(f"Testing {len(weights)} physics weights: {weights}")
    
    for i, w in enumerate(weights):
        print(f"\n{'='*70}")
        print(f"PROGRESS: {i+1}/{len(weights)} - Testing physics weight: {w}")
        print(f"{'='*70}")
        
        try:
            result = run_search(w, X_train, X_test, y_train, y_test, layers_module)
            all_results.append(result)
            results[f"w_{w}"] = result
            
            # Track best result
            if result['test_loss'] < best_loss:
                best_loss = result['test_loss']
                best_result = result
                print(f"NEW BEST RESULT! Test loss: {best_loss:.6f}")
            
            print(f"Completed {w} successfully")
            
        except Exception as e:
            print(f"ERROR with physics weight {w}: {e}")
            # Store error info
            error_result = {
                'name': 'ERROR',
                'test_loss': float('inf'),
                'test_emp_loss': float('inf'),
                'test_phy_loss': float('inf'),
                'MAD': float('inf'),
                'error': str(e),
                'hyperparameters': {'physics_weight': w, 'epochs': 0}
            }
            all_results.append(error_result)
            continue

    print(f"\n{'='*70}")
    print("FINAL GRID SEARCH RESULTS")
    print(f"{'='*70}")

    if best_result and best_result['name'] != 'ERROR':
        print("BEST CONFIGURATION:")
        print(f"   Physics weight: {best_result['hyperparameters']['physics_weight']}")
        print(f"   Test loss: {best_result['test_loss']:.6f}")
        print(f"   Test empirical loss: {best_result['test_emp_loss']:.6f}")
        print(f"   Test physics loss: {best_result['test_phy_loss']:.6f}")
        print(f"   MAD: {best_result['MAD']:.6f}")
        print(f"   Epochs completed: {best_result['hyperparameters']['epochs']}")
        print(f"   Training time: {best_result['training_time']:.1f}s")
        
        print("\nCOMPLETE RESULTS RANKING:")
        # Sort results by test loss
        successful_results = [r for r in all_results if r['name'] != 'ERROR']
        successful_results.sort(key=lambda x: x['test_loss'])
        
        # Print table header
        print(f"   {'Rank':<4} {'Weight':<8} {'Loss':<10} {'Emp.Loss':<10} {'Phy.Loss':<10} {'MAD':<10} {'Epochs':<6} {'Time(s)':<8}")
        print(f"   {'-'*4} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*8}")
        
        # Print each result
        for rank, result in enumerate(successful_results, 1):
            pw = result['hyperparameters']['physics_weight']
            tl = result['test_loss']
            emp_loss = result['test_emp_loss']
            phy_loss = result['test_phy_loss']
            mad = result['MAD']
            ep = result['hyperparameters']['epochs']
            tt = result['training_time']
            print(f"   {rank:<4} {pw:<8.4f} {tl:<10.6f} {emp_loss:<10.6f} {phy_loss:<10.6f} {mad:<10.6f} {ep:<6d} {tt:<8.1f}")
        
        # Show failed runs
        failed_results = [r for r in all_results if r['name'] == 'ERROR']
        if failed_results:
            print("\nFAILED RUNS:")
            for result in failed_results:
                pw = result['hyperparameters']['physics_weight']
                error = result.get('error', 'Unknown error')
                print(f"   Weight={pw}: {error}")
        
        # Create comparison plot if multiple successful results
        if len(successful_results) > 1:
            create_comparison_plots(successful_results, best_result)
    else:
        print("NO SUCCESSFUL RESULTS FOUND!")
        print("All runs failed. Check your model and data.")

    return best_result, all_results, results

def create_comparison_plots(successful_results, best_result):
    plt.figure(figsize=(15, 12))  # Increased height to accommodate new plot
    
    # Loss comparison
    plt.subplot(3, 3, 1)
    weights_plot = [r['hyperparameters']['physics_weight'] for r in successful_results]
    losses_plot = [r['test_loss'] for r in successful_results]
    plt.plot(weights_plot, losses_plot, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Physics Weight')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Physics Weight')
    plt.grid(True, alpha=0.3)
    
    # Empirical Loss comparison
    plt.subplot(3, 3, 2)
    emp_losses_plot = [r['test_emp_loss'] for r in successful_results]
    plt.plot(weights_plot, emp_losses_plot, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Physics Weight')
    plt.ylabel('Empirical Loss')
    plt.title('Empirical Loss vs Physics Weight')
    plt.grid(True, alpha=0.3)
    
    # Physics Loss comparison
    plt.subplot(3, 3, 3)
    phy_losses_plot = [r['test_phy_loss'] for r in successful_results]
    plt.plot(weights_plot, phy_losses_plot, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Physics Weight')
    plt.ylabel('Physics Loss')
    plt.title('Physics Loss vs Physics Weight')
    plt.grid(True, alpha=0.3)
    
    # MAD comparison
    plt.subplot(3, 3, 4)
    mad_plot = [r['MAD'] for r in successful_results]
    plt.plot(weights_plot, mad_plot, 'mo-', linewidth=2, markersize=8)
    plt.xlabel('Physics Weight')
    plt.ylabel('MAD')
    plt.title('MAD vs Physics Weight')
    plt.grid(True, alpha=0.3)
    
    # Training time comparison
    plt.subplot(3, 3, 5)
    time_plot = [r['training_time']/60 for r in successful_results]  # Convert to minutes
    plt.plot(weights_plot, time_plot, 'co-', linewidth=2, markersize=8)
    plt.xlabel('Physics Weight')
    plt.ylabel('Training Time (minutes)')
    plt.title('Training Time vs Physics Weight')
    plt.grid(True, alpha=0.3)
    
    # Best model loss curves
    plt.subplot(3, 3, 6)
    if 'all_losses' in best_result:
        plt.plot(best_result['all_losses'], 'b-', label='Total', linewidth=2)
        plt.plot(best_result['empirical_losses'], 'r-', label='Empirical', linewidth=2)
        plt.plot(best_result['physical_losses'], 'g-', label='Physics', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Best Model Loss Curves\n(Weight: {best_result["hyperparameters"]["physics_weight"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Best model predictions
    plt.subplot(3, 3, 7)
    plt.scatter(best_result['y_test'], best_result['y_pred_test'], alpha=0.7, s=50)
    min_val = min(best_result['y_test'].min(), best_result['y_pred_test'].min())
    max_val = max(best_result['y_test'].max(), best_result['y_pred_test'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Best Model: True vs Predicted\n(MAD: {best_result["MAD"]:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Physics vs Empirical Loss scatter plot
    plt.subplot(3, 3, 8)
    emp_values = [r['test_emp_loss'] for r in successful_results]
    phy_values = [r['test_phy_loss'] for r in successful_results]
    plt.scatter(emp_values, phy_values, alpha=0.7, s=80, c='purple')
    
    # Add weight labels to each point
    for i, weight in enumerate(weights_plot):
        plt.annotate(f"{weight:.4f}", 
                     (emp_values[i], phy_values[i]),
                     xytext=(5, 5),
                     textcoords="offset points")
    
    plt.xlabel('Empirical Loss')
    plt.ylabel('Physics Loss')
    plt.title('Empirical vs Physics Loss')
    plt.grid(True, alpha=0.3)
    
    # Performance summary
    plt.subplot(3, 3, 9)
    components = ['Test Loss', 'Emp. Loss', 'Phy. Loss', 'MAD', 'Time (min)']
    values = [
        best_result['test_loss'], 
        best_result['test_emp_loss'],
        best_result['test_phy_loss'],
        best_result['MAD'], 
        best_result['training_time']/60
    ]
    colors = ['blue', 'red', 'green', 'magenta', 'cyan']
    bars = plt.bar(components, values, color=colors, alpha=0.7)
    plt.title('Best Model Metrics')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('grid_search_comparison.png')
    plt.close()

# Two-step gradient debugging function
def debug_gradients(X_train, y_train, layers_module):
    print("=== Two-step gradient debug (Adam only) ===")

    # Fresh small model and data copies
    _dbg_m = PGGCN_Hybrid(layers_module)
    _dbg_m.addRule("sum", 0, 32)
    _dbg_m.addRule("multiply", 32, 33)
    _dbg_m.addRule("distance", 33, 36)

    # Prepare one-batch dataset
    _dbg_X = [np.copy(x) for x in X_train]
    for i in range(len(_dbg_X)):
        if _dbg_X[i].shape[0] < 2000:
            _dbg_X[i] = np.concatenate([_dbg_X[i], np.zeros([2000 - _dbg_X[i].shape[0], 53])], axis=0)
    _dbg_X = np.array(_dbg_X).astype(np.float32)
    _dbg_y = np.array(y_train).astype(np.float32)

    _dbg_m.set_input_shapes([x.shape[0] for x in _dbg_X])
    _dbg_opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

    # Step 0: forward only checks
    with tf.GradientTape(persistent=True) as tape0:
        tape0.watch(_dbg_m.trainable_variables)
        preds0 = _dbg_m(_dbg_X, training=True)
        emp0 = pure_rmse_hybrid(_dbg_y, preds0[:,0])
        phy0 = physical_consistency_loss(_dbg_y, preds0[:,0], preds0[:,1:16])
        tot0 = emp0 + 0.001*phy0

    print("preds0 shape:", preds0.shape, "dtype:", preds0.dtype)
    print("emp0:", float(emp0.numpy()), "phy0:", float(phy0.numpy()), "tot0:", float(tot0.numpy()))
    print("has_nan preds:", bool(tf.reduce_any(tf.math.is_nan(preds0))))
    print("has_inf preds:", bool(tf.reduce_any(tf.math.is_inf(preds0))))

    # Gradients at step 0
    grads0 = tape0.gradient([emp0, 0.001*phy0], _dbg_m.trainable_variables)
    if isinstance(grads0[0], (list, tuple)):
        _gflat = []
        for g in grads0:
            _gflat.append(g)
        grads0 = _gflat
    report_none_grads(grads0, _dbg_m.trainable_variables, "step0")

    # Apply one update with Adam to see if variables break gradient flow
    _dbg_opt.apply_gradients([(g, v) for g, v in zip(grads0, _dbg_m.trainable_variables) if g is not None])

    # Step 1: recompute on updated weights
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(_dbg_m.trainable_variables)
        preds1 = _dbg_m(_dbg_X, training=True)
        emp1 = pure_rmse_hybrid(_dbg_y, preds1[:,0])
        phy1 = physical_consistency_loss(_dbg_y, preds1[:,0], preds1[:,1:16])
        tot1 = emp1 + 0.001*phy1

    print("preds1 shape:", preds1.shape, "dtype:", preds1.dtype)
    print("emp1:", float(emp1.numpy()), "phy1:", float(phy1.numpy()), "tot1:", float(tot1.numpy()))
    print("has_nan preds1:", bool(tf.reduce_any(tf.math.is_nan(preds1))))
    print("has_inf preds1:", bool(tf.reduce_any(tf.math.is_inf(preds1))))

    # Gradients at step 1
    grads1 = tape1.gradient([emp1, 0.001*phy1], _dbg_m.trainable_variables)
    report_none_grads(grads1, _dbg_m.trainable_variables, "step1")

    print("=== End two-step gradient debug ===")
    return

# Main function
def main():
    # Load data
    print("Loading data...")
    PDBs, df = load_data()
    
    # Import custom modules
    print("Importing custom modules...")
    layers_module, get_atom_features = import_custom_modules()
    
    # Preprocess data
    print("Preprocessing data...")
    info = extract_info(PDBs, df)
    X, y, X_train, X_test, y_train, y_test = prepare_data(PDBs, df, info, get_atom_features)
    
    print(f"Dataset sizes: {len(X_train)} training, {len(X_test)} testing")
    
    # # Debug gradients
    # debug_gradients(X_train, y_train, layers_module)
    
    # Run grid search
    print("STARTING OPTIMIZED GRID SEARCH")
    print("=" * 70)
    weights = [0.01, 0.05, 0.1, 0.5, 1.0]
    best_result, all_results, results = run_full_grid_search(
        X_train, X_test, y_train, y_test, layers_module, weights)
    
    # Save results
    with open('grid_search_results.pkl', 'wb') as f:
        pickle.dump({
            'best_result': best_result,
            'all_results': all_results,
            'results': results
        }, f)
    
    print("Results saved to grid_search_results.pkl")
    print("Script execution completed!")

if __name__ == "__main__":
    main()

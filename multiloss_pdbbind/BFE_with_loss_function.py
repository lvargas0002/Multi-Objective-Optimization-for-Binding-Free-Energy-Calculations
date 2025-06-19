import pandas as pd
import tensorflow as tf
import numpy as np
import os
import pickle
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers, constraints, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import importlib

# Import layers module
import models.layers_update_mobley as layers
importlib.reload(layers)

# Define helper functions for loss calculation
def pure_rmse(y_true, y_pred):
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


def combined_loss(physics_hyperparam=0.0003):
    def loss_function(y_true, y_pred):
        # Extract prediction and physics info
        prediction = y_pred[:, 0]
        physics_info = y_pred[:, 1:16]  # Assuming 15 physical features
        
        # Calculate individual loss components
        empirical_loss = pure_rmse(y_true, prediction)
        physics_loss = physical_consistency_loss(y_true, prediction, physics_info)
        
        # Combine losses with weights
        total_loss = empirical_loss + (physics_hyperparam * physics_loss)
        
        return total_loss
    
    return loss_function

# Callback to track loss components
class LossComponentsCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_instance):
        super().__init__()
        self.empirical_losses = []
        self.physical_losses = []
        self.total_losses = []
        self.learning_rates = []
        self.model = model_instance
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Store the total loss
        self.total_losses.append(logs.get('loss'))
        

        # Store learning rate
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)  # Call the schedule
        else:
            lr = lr  

        try:
            self.learning_rates.append(float(tf.keras.backend.get_value(lr)))
        except:
            # If conversion fails, use previous value or 0
            if self.learning_rates:
                self.learning_rates.append(self.learning_rates[-1])
            else:
                self.learning_rates.append(0.0)

# Model definition
class PGGCNModel(tf.keras.Model):
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=1024, l2=1e-4, dropout_rate=0.2, maxnorm=3.0):
        super().__init__()
        self.ruleGraphConvLayer = layers.RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.ruleGraphConvLayer.combination_rules = []
        self.conv = layers.ConvLayer(c_out_channel, r_out_channel)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', name='dense1', 
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense5 = tf.keras.layers.Dense(16, activation='relu', name='dense2', 
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense6 = tf.keras.layers.Dense(1, name='dense6', 
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dense7 = tf.keras.layers.Dense(1, name='dense7',
                                           kernel_initializer=tf.keras.initializers.Constant([.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]),
                                           bias_initializer=tf.keras.initializers.Zeros(),
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))

    def addRule(self, rule, start_index, end_index=None):
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)
        
    def set_input_shapes(self, i_s):
        self.i_s = i_s

    def call(self, inputs, training=True):
        physics_info = inputs[:, 0, 38:] 
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

# Check if the model_results directory exists, if not create it
save_dir = "model_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Main script execution
if __name__ == "__main__":
    # Read data files
    df = pd.read_csv('Datasets/pdbbind_100.csv')
    PDBs = pickle.load(open('Datasets/PDBBind_100.pkl', 'rb'))
    
    # Filter data
    df = df[df['complex-name'] != '1.00E+66']
    pdb_keys = set(PDBs.keys())
    df_filtered = df[df['complex-name'].isin(pdb_keys)]
    
    # Select relevant columns
    df_final = df_filtered[['complex-name','pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals', 
                          'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
                          'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec', 
                          'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb', 
                          'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf','ddg']]
    
    # Extract feature information
    info = []
    for pdb in list(PDBs.keys()):
        info.append(df_final[df_final['complex-name'] == pdb][['pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals', 
                                                             'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
                                                             'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec', 
                                                             'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb', 
                                                             'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf']].to_numpy()[0])
    
    # Define featurization function
    from models.dcFeaturizer import atom_features as get_atom_features
    
    def featurize(molecule, info):
        atom_features = []
        for atom in molecule.GetAtoms():
            new_feature = get_atom_features(atom).tolist()
            position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
            new_feature += [atom.GetMass(), atom.GetAtomicNum(),atom.GetFormalCharge()]
            new_feature += [position.x, position.y, position.z]
            for neighbor in atom.GetNeighbors()[:2]:
                neighbor_idx = neighbor.GetIdx()
                new_feature += [neighbor_idx]
            for i in range(2 - len(atom.GetNeighbors())):
                new_feature += [-1]

            atom_features.append(np.concatenate([new_feature, info], 0))
        return np.array(atom_features)
    
    # Featurize molecules
    X = []
    y = []
    for i, pdb in enumerate(list(PDBs.keys())):
        X.append(featurize(PDBs[pdb], info[i]))
        y.append(df_final[df_final['complex-name'] == pdb]['ddg'].to_numpy()[0])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    
    # Train model with physics loss
    # Define hyperparameters
    physics_hyperparam = [0.000005]
    epochs = [500]
    lr_schedule = ExponentialDecay(
            initial_learning_rate=0.005,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True
        )
    
    # Initialize lists to store results
    y_differences = []
    total_losses = []
    empirical_losses = []
    physics_losses = []
    all_results = []
    
    # Training loop
    for epoch in epochs:
        for physics_weight in physics_hyperparam:
            print("---------- Hyperparameter combinations ------------")
            print("Epoch : {};  physics_weight: {};".format(str(epoch), str(physics_weight)))
            
            # Initialize model
            m = PGGCNModel()
            m.addRule("sum", 0, 32)
            m.addRule("multiply", 32, 33)
            m.addRule("distance", 33, 36)
            
            # Compile model
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            m.compile(loss=combined_loss(physics_weight), optimizer=opt)
            
            # Prepare data
            input_shapes = []
            for i in range(len(X_train)):
                input_shapes.append(np.array(X_train[i]).shape[0])
            m.set_input_shapes(input_shapes)
            
            for i in range(len(X_train)):
                if X_train[i].shape[0] < 12000:
                    new_list = np.zeros([12000 - X_train[i].shape[0], 53])
                    X_train[i] = np.concatenate([X_train[i], new_list], 0)
            
            X_train_arr = np.array(X_train)
            y_train_arr = np.array(y_train)
            
            # Initialize loss tracker
            loss_tracker = LossComponentsCallback(m)
            
            # Add early stopping
            early_stopping = EarlyStopping(
                monitor='loss',           
                patience=20,              
                restore_best_weights=True, 
                min_delta=0.001,          
                verbose=1                 
            )
            
            # Train model
            hist = m.fit(X_train_arr, y_train_arr, epochs=epoch, batch_size=len(X_train_arr), 
                         callbacks=[loss_tracker, early_stopping])
            
            # Evaluate on test set
            input_shapes = []
            for i in range(len(X_test)):
                input_shapes.append(np.array(X_test[i]).shape[0])
            m.set_input_shapes(input_shapes)
            
            for i in range(len(X_test)):
                if X_test[i].shape[0] < 12000:
                    new_list = np.zeros([12000 - X_test[i].shape[0], 53])
                    X_test[i] = np.concatenate([X_test[i], new_list], 0)
            
            X_test_arr = np.array(X_test)
            x_c = copy.deepcopy(X_test_arr)
            y_test_arr = np.array(y_test)
            
            # Get predictions
            y_pred_test = m.predict(X_test_arr)
            y_pred_test = np.array(y_pred_test[:, 0])
            
            # Calculate metrics
            y_difference = np.mean(np.abs(np.abs(y_test_arr) - np.abs(y_pred_test)))
            eval_loss = m.evaluate(X_test_arr, y_test_arr)
            print("The mean absolute difference between y_tru & y_pred is : {}".format(str(y_difference)))
            
            # Save the final training loss
            final_train_loss = loss_tracker.total_losses[-1] if loss_tracker.total_losses else None
            
            # Save results
            training_data = {
                "total_losses": loss_tracker.total_losses,
                "empirical_losses": loss_tracker.empirical_losses,
                "physical_losses": loss_tracker.physical_losses,
                "learning_rates": loss_tracker.learning_rates,
                "epochs": list(range(1, len(loss_tracker.total_losses) + 1)),
                "physics_hyperparam": physics_weight
            }
            
            with open(os.path.join(save_dir, "hybrid_training_data.pkl"), "wb") as f:
                pickle.dump(training_data, f)
            
            test_results = {
                "predictions": y_pred_test,
                "actual_values": y_test_arr,
                "mean_abs_diff": y_difference,
                "test_loss": eval_loss
            }
            
            with open(os.path.join(save_dir, "hybrid_test_results.pkl"), "wb") as f:
                pickle.dump(test_results, f)
            
            print(f"Results saved to {save_dir}")
            
            # Plot loss components
            plt.figure(figsize=(12, 8))
            epoch_length = range(1, len(loss_tracker.total_losses) + 1)
            
            # Total loss
            plt.subplot(2, 2, 1)
            plt.plot(epoch_length, loss_tracker.total_losses, 'b-', label='Total Loss')
            plt.title('Total Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            # Can uncomment these if needed
            # # Learning rate
            # plt.subplot(2, 2, 2)
            # plt.plot(epochs, loss_tracker.learning_rates, 'g-', label='Learning Rate')
            # plt.title('Learning Rate')
            # plt.xlabel('Epochs')
            # plt.ylabel('Learning Rate')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig('loss_components.png')
            # plt.show()
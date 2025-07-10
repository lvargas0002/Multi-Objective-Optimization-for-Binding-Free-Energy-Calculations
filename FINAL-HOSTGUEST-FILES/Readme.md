# Binding Free Energy Calculation (ΔΔG) Notebooks:
1. `BFE_LossFunction_submitted.ipynb` : This notebook implements a machine learning pipeline for predicting binding free energy (ΔΔG) using a hybrid physics-guided rule-based graph convolutional neural network (PG-RGCN) model with a multi-objective loss function. The workflow integrates molecular feature extraction, data preprocessing, model definition, training, evaluation, and visualization, with a focus on incorporating both empirical and physics-based loss components to enhance model physical consistency.

 ### Main Features
      1. Data Preparation: 
      •	Loads molecular structure data and experimental ΔΔG values. The dataset in present in the subidrectory - Datasets.
      •	Featurizes molecules using RDKit and custom graph-based descriptors.
      •	Splits data into training and testing sets.
      
      2. Model Architecture:
      •	Defines a custom hybrid GCN model (PGGCN_Hybrid) using TensorFlow/Keras. The model files used in this part are present in the subdirectory - models.
      •	Incorporates specialized graph convolutional layers and dense layers.
      •	Integrates physical energy terms as additional input features.

      3. Loss Function:
      Implements a combined loss function:
          •	Empirical Loss: Standard RMSE between predicted and true ΔΔG.
          •	Structural Loss: To mitigate overfitting of the model, we use L2 regularization and dropout layers within the model. 
            These components serve to penalize overly complex models and prevent overfitting by regularizing the weights of the neural network layers.
          •	Physical Consistency Loss: Penalizes deviations from physically calculated ΔΔG using molecular energy components.
          •	Weighted sum of both losses, with adjustable hyperparameters.

      4. Training Pipeline:
      •	Supports hyperparameter tuning (epochs, physics loss weight, learning rate schedule).
      •	Includes early stopping based on loss convergence.
      •	Tracks and logs total, empirical, and physical loss components per epoch.

      5. Evaluation & Visualization:
      •	Evaluates model performance on the test set, reporting mean absolute differences. The results are stored in the subdirectory - Result_Pickle.
      •	Plots loss curves over epochs for total, empirical, and physical losses. The graphs are saved in the subdirectory for Graphs.
      •	Generates scatter plots comparing predicted vs. experimental ΔΔG.

      6. Baseline Comparison:
      •	Includes a section for training and evaluating the model without the physics-based loss for direct comparison.

### Usage
•	Dependencies: Requires TensorFlow, Keras, RDKit, DeepChem, NumPy, pandas, matplotlib, and scikit-learn.

•	Input Data: Molecular structure files and CSVs with experimental binding free energy and energy decomposition terms.

•	Customization: Users can adjust model hyperparameters, loss weights, and feature engineering steps as needed.


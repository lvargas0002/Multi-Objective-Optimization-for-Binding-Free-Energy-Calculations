This directory contains scripts for performing k-fold cross-validation in the binding free energy prediction project.
### Purpose
K-fold cross-validation is used to assess the generalizability and robustness of the machine learning models for binding free energy and entropy predictions. 
By training and validating on different data splits, this approach provides a more reliable estimate of model performance.

### Contents
- `k-fold-bfe-runner.py` - Runner script to execute k-fold cross-validation runs without the inclusion of multi-objective loss function for DDG.
- `k-fold-bfe-multiloss-runner.py` – Main runner script to execute k-fold cross-validation runs with the inclusion of multi-objective loss function for DDG.
- `k-fold-nma-runner.py` - Runner script to execute k-fold cross-validation runs without the inclusion of multi-objective loss function for TDS with data from NMA simulations.
- `k-fold-nma-multiloss-runner.py` - Main runner script to execute k-fold cross-validation runs with the inclusion of multi-objective loss function for TDS with data from NMA simulations.
- `k-fold-vm2-multiloss-runner.py` - Runner script to execute k-fold cross-validation runs with the inclusion of multi-objective loss function for TDS with data from VM2 simulations taken from previous PRGCN model. (Just for comparison, not required for this paper).
- `pgcn-ddg-run.sh` - Bash script to run the runner files on curie server. 

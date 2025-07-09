# Multi Objective Loss Function for Free Energy Calculations
### About Project
Accurate estimation of binding free energy (ΔΔ𝐺) remains a critical challenge in drug discovery, directly influencing the development of effective therapeutics. This project aims to predict the binding free energy, which measures the thermodynamic favorability of a ligand binding to a biological target such as a protein or receptor.  Many deep learning models, particularly graph convolutional networks (GCNs), have demonstrated the ability to rapidly predict molecular properties by learning hierarchical representations from large-scale chemical data, yet frequently lack explicit incorporation of physical laws, leading to potential issues with interpretability and generalizability. While reducing prediction error is necessary, it is not sufficient: models must also respect physical constraints (e.g., thermodynamic consistency) and maintain interpretability and simplicity to avoid overfitting. Hybrid physics-guided graph convolutional neural networks (PGCNs) used in this project seek to overcome these limitations by embedding physically meaningful features and constraints within deep learning architectures. In this study, we introduce a multi-objective loss function in PGCN model that simultaneously optimizes empirical error, structural similarity, and physical consistency by integrating molecular fingerprints with physics-based features in a unified GCN framework. The main objective of this paper is to create a more accurate and efficient pipeline for predicting binding free energy  and entropy while ensuring the results remain consistent with physics-based simulations.

# CONTENTS OF THIS DIRECTORY
This directory and it's subdirectories provides below content:
### Data_Collection 
This folder has the Jupyter Notebook file used for data collection of Host-Guest systems, along with the data in .csv files. For more Details about the cd-set1, cd-set2 and gdcc systems, refer to Mobley GitHub page [1].
### FINAL-HOSTGUEST-FILES
This folder consists of all the files for calculation of Binding Free Energy Calculations, i.e Delta G and Delta S calculations, with and without the addition of the Multi-objective Loss Function, the graphs and tables generated to compare the data and results.
### multiloss_pdbbind
This folder contains Notebook files that is used for transfer learning on larger datasets like PDBbind. These files are work in progress.

# References:
1. Mobley's lab: https://github.com/MobleyLab/benchmarksets



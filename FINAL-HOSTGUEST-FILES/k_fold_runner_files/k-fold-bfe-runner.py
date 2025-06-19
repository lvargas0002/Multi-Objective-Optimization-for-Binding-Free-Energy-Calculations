import pandas as pd
import tensorflow as tf
import numpy as np
import os
import conda_installer
from rdkit import Chem
from deepchem.feat.graph_features import atom_features as get_atom_features
import rdkit
import pickle
import matplotlib.pyplot as plt
from models.PGCN_model_bfe import get_trained_model, test_model

X_folds= pickle.load(open('/home/lthoma21/BFE-Loss-Function/FINAL-HOSTGUEST-FILES/Datasets/X_folds_BFE_4.pkl', 'rb'))
y_folds = pickle.load(open('/home/lthoma21/BFE-Loss-Function/FINAL-HOSTGUEST-FILES/Datasets/y_folds_BFE_4.pkl', 'rb'))

K = 4
epoch = 200
hists = []
test_loss = []
predicts_train = []
predicts_test = []
y_diff = []
y_tests=[]

for k in range(K):
    X_train = []
    for i in range(K):
        if i != k:
            X_train += X_folds[i]
    X_test = X_folds[k]
    y_train = np.concatenate(np.array([y_folds[i] for i in range(K) if i != k]), 0)
    y_test = np.array(y_folds[k])
    #  Passing the X_train and x_add_train to train the model.
    hs, m, x_converted = get_trained_model(X_train, y_train, epochs = epoch, max_num_atoms = 2000, n_features = 53)
    with open('/home/lthoma21/BFE-Loss-Function/FINAL-HOSTGUEST-FILES/Datasets/PGCN_bfe_without_multiloss_K%d_W_n.pkl' % k, 'wb') as f:
        pickle.dump([m.layers[0].w_n, m.layers[0].w_n], f)
        f.close()
    predicts_train.append(m.predict(x_converted, batch_size=len(X_train)))
    hists.append(hs)
    y_tests.append(y_test)
    k_loss,y_difference, x_converted = test_model(X_test, y_test, m, max_num_atoms = 2000, n_features = 53)
    test_loss.append(k_loss)
    y_diff.append(y_difference)
    predicts_test.append(m.predict(x_converted, batch_size=len(X_test)))
    

with open('/home/lthoma21/BFE-Loss-Function/FINAL-HOSTGUEST-FILES/Datasets/PGCN_BFE_K%d_fold_hists.pkl' % K, 'wb') as file:
    pickle.dump(hists, file)
with open('/home/lthoma21/BFE-Loss-Function/FINAL-HOSTGUEST-FILES/Datasets/PGCN_BFE_K%d_fold_test.pkl' % K, 'wb') as file:
    pickle.dump(test_loss, file)
with open('/home/lthoma21/BFE-Loss-Function/FINAL-HOSTGUEST-FILES/Datasets/PGCN_BFE_K%d_fold_train_pred.pkl' % K, 'wb') as file:
    pickle.dump(predicts_train, file)
with open('/home/lthoma21/BFE-Loss-Function/FINAL-HOSTGUEST-FILES/Datasets/PGCN_BFE_K%d_fold_test_pred.pkl' % K, 'wb') as file:
    pickle.dump(predicts_test, file)
with open('/home/lthoma21/BFE-Loss-Function/FINAL-HOSTGUEST-FILES/Datasets/PGCN_BFE_K%d_fold_y_test.pkl' % K, 'wb') as file:
    pickle.dump(y_tests, file)
with open('/home/lthoma21/BFE-Loss-Function/FINAL-HOSTGUEST-FILES/Datasets/PGCN_BFE_K%d_fold_y_diff.pkl' % K, 'wb') as file:
    pickle.dump(y_diff, file)
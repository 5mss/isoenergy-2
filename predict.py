import h5py
import time
import numpy as np
from sklearn.externals import joblib

n_sample = 900
N = 1005  # length of feature data
N2 = N*N
Nt = 201
print('Loading input data...')
start = time.time()
with h5py.File('target2_.h5', 'r') as xp:  # loading problem data
    X_pred = xp['QPI'][...]

# Nt = int(len(X_pred) / 10)
# X_pred = X_pred[0:Nt]
print('Loading models...')  # loading models
Model = joblib.load('testModel0')
pca_target = joblib.load('PCA_target_1500')
scaler_target = joblib.load('Scaler_target')
print('Predicting...')  # predicting
Y_pred = Model.predict(X_pred)
print('Recovering...')  # recover
# out_scaled = pca_target.inverse_transform(Y_pred)
# out = scaler_target.inverse_transform(out_scaled)
print('Saving...')
n = len(Y_pred)
with h5py.File('answer_polar_class2.h5', 'w') as opt:  # save output data
    for i in range(n):
        opt.create_group(f'{i:04d}')
        opt[f'{i:04d}']['isoE'] = Y_pred[i]
            # .reshape((Nt, Nt))

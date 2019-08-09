import h5py
import time
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import preprocessing

n_sample = 900
N = 1005  # length of feature data
N2 = N*N
Nt = 201
print('Loading input data...')
start = time.time()
data = np.empty((n_sample, N2), dtype=float)
with h5py.File('problem.h5', 'r') as ipt:
    for i in range(900):
        sample = ipt[f'{i:04d}']['QPI'][...]
        data[i] = sample.reshape((1, N2))
print('Loading complete. Time used: ', time.time() - start)
print('Scaleing input data...')
start = time.time()
scaler_feature = joblib.load('Scaler_feature')
data_scaled = scaler_feature.transform(data)
print('Scaling complete. Time used: ', time.time() - start)
print('Feature extracting...')
start = time.time()
pca_feature = joblib.load('PCA')
X_pred = pca_feature.transform(data_scaled)
print('Feature extracting complete. Time used: ', time.time() - start)
print('Saving predict data...')
start = time.time()
with h5py.File('X_pred.h5', 'w') as xp:
    xp['feature'] = X_pred
print('Saving complete. Time used: ', time.time() - start)
print('Loading models...')
nnModel = joblib.load('lsModel')
pca_target = joblib.load('PCA_target')
scaler_target = joblib.load('Scaler_target')
print('Predicting...')
Y_pred = nnModel.predict(X_pred)
print('Recovering...')
out_scaled = pca_target.inverse_transform(Y_pred)
out = scaler_target.inverse_transform(out_scaled)
print('Saving...')
with h5py.File('answer_ls.h5', 'w') as opt:
    for i in range(900):
        opt.create_group(f'{i:04d}')
        opt[f'{i:04d}']['isoE'] = out[i].reshape((Nt, Nt))

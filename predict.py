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
data = np.empty((n_sample, N2), dtype=float)
with h5py.File('problem.h5', 'r') as ipt:  # load data
    for i in range(900):
        sample = ipt[f'{i:04d}']['QPI'][...]
        data[i] = sample.reshape((1, N2))
print('Loading complete. Time used: ', time.time() - start)
print('Scaleing input data...')
start = time.time()
scaler_feature = joblib.load('Scaler_feature')  # scale input data
data_scaled = scaler_feature.transform(data)
print('Scaling complete. Time used: ', time.time() - start)
print('Feature extracting...')  # feature extract
start = time.time()
pca_feature = joblib.load('PCA')
X_pred = pca_feature.transform(data_scaled)
print('Feature extracting complete. Time used: ', time.time() - start)
print('Saving predict data...')
start = time.time()
with h5py.File('X_pred.h5', 'w') as xp:  # saving preprocessed input
    xp['feature'] = X_pred
print('Saving complete. Time used: ', time.time() - start)
print('Loading models...')  # loading models
nnModel = joblib.load('krModel_20')
pca_target = joblib.load('PCA_target_20')
scaler_target = joblib.load('Scaler_target')
print('Predicting...') # predicting
Y_pred = nnModel.predict(X_pred)
print('Recovering...')  # recover
out_scaled = pca_target.inverse_transform(Y_pred)
out = scaler_target.inverse_transform(out_scaled)
print('Saving...')
with h5py.File('answer_kr_20.h5', 'w') as opt: # save output data
    for i in range(900):
        opt.create_group(f'{i:04d}')
        opt[f'{i:04d}']['isoE'] = out[i].reshape((Nt, Nt))

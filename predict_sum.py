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
with h5py.File('X_pred.h5', 'r') as xp:
    X_pred = xp['feature'][...]
print('Loading models...')
nnModel = joblib.load('nnModel_sum')
scaler_target = joblib.load('Scaler_sum')
print('Predicting...')
Y_pred = nnModel.predict(X_pred)
print('Recovering...')
A = scaler_target.inverse_transform(Y_pred)
print('Saving...')
with h5py.File('answer_sum.h5', 'w') as opt:
    with h5py.File('problem.h5', 'r') as ipt:
        for i in range(900):
            D = ipt[f'{i:04d}']['QPI'][...]
            B = np.sqrt((D - A[i] ** 3) / 3 * A[i])
            B = np.nan_to_num(B)
            f = np.fft.ifft2(B, [Nt, Nt])
            f = np.abs(f)
            opt.create_group(f'{i:04d}')
            opt[f'{i:04d}']['isoE'] = f

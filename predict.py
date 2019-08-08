import h5py
import time
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import preprocessing

n_sample = 900
N = 1005  # length of feature data
N2 = N*N
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
scaler = preprocessing.StandardScaler(copy=False)
scaler.fit(data)
data_scaled = scaler.transform(data)
print('Scaling complete. Time used: ', time.time() - start)

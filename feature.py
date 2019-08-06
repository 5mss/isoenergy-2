import h5py
import time
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.externals import joblib
N = 1005
N2 = N*N
n = 1000
nComp = n
part = 0
data = np.empty((n, N2), dtype=float)
start = time.time()
print('Loading PCA model...')
pca = joblib.load('PCA')
print('Loading PCA complete. Time used: ', time.time() - start)
start = time.time()
scaler = preprocessing.StandardScaler()
with h5py.File('train.h5', 'r') as ipt:
    while n <= 9000:
        for i in range(n-1000, n):
            print(f'Loading data part {part}...')
            start = time.time()
            sample = ipt[f'{i:04d}']['QPI'][...]
            sample = sample.reshape((1, N2))
            data[i] = sample
            print('Loading complete. Time used: ', time.time() - start)
        print(f'Scaling data part {part}...')
        start = time.time()
        scaler.fit(data)
        data_scaled = scaler.transform(data)
        print('Scaling complete. Time used: ', time.time() - start)
        print(f'Feature extracting with PCA part {part}...')
        start = time.time()
        data_reduced = pca.transform(data_scaled)
        print('Feature extracting complete. Time used: ', time.time() - start)
        start = time.time()
        print(f'Saving reduced data part {part}')
        n += 1000
        part += 1

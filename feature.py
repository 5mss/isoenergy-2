import h5py
import time
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
N = 1005
N2 = N*N
n = 2000
part_size = 1000
part = 1
data = np.empty((part_size, N2), dtype=float)
start = time.time()
print('Loading PCA model...')
pca = joblib.load('PCA')
print('Loading PCA complete. Time used: ', time.time() - start)
start = time.time()
scaler = preprocessing.StandardScaler(copy=False)
while n <= 9000:
    print(f'Loading data part {part}...')
    start = time.time()
    with h5py.File('train.h5', 'r') as ipt:
        for i in range(n-1000, n):
            sample = ipt[f'{i:04d}']['QPI'][...]
            data[i-part * 1000] = sample.reshape((1, N2))
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
    print(f'Saving reduced data part {part}...')
    start = time.time()
    with h5py.File('train_feature.h5', 'w') as opt:
        opt.create_group('/feature')
        opt['feature'][f'{part}'] = data_reduced
    print(f'Complete saving. Time used: ', time.time() - start)
    n += 1000
    part += 1

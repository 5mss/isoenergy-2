import h5py
import time
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
N = 201  # length of target data
N2 = N*N
n = 1000  # current line
part_size = 1000  # samples per part
part = 0  # current part
data = np.empty((part_size, N2), dtype=float)
start = time.time()
print('Loading PCA model...')
pca = joblib.load('PCA_target_20')
print('Loading PCA complete. Time used: ', time.time() - start)
start = time.time()
scaler = joblib.load('Scaler_target')
with h5py.File('train_target_20.h5', 'w') as opt:
    opt.create_group('/target')
    while n <= 9000:
        print(f'Loading data part {part}...')
        start = time.time()
        with h5py.File('train.h5', 'r') as ipt:
            for i in range(n-1000, n):
                sample = ipt[f'{i:04d}']['isoE'][...]
                data[i-part * 1000] = sample.reshape((1, N2))
        print('Loading complete. Time used: ', time.time() - start)
        print(f'Scaling data part {part}...')
        start = time.time()
        data_scaled = scaler.transform(data)
        print('Scaling complete. Time used: ', time.time() - start)
        print(f'Feature extracting with PCA part {part}...')
        start = time.time()
        data_reduced = pca.transform(data_scaled)
        print('Feature extracting complete. Time used: ', time.time() - start)
        print(f'Saving reduced data part {part}...')
        start = time.time()
        opt['target'][f'{part}'] = data_reduced
        print(f'Complete saving. Time used: ', time.time() - start)
        n += 1000
        part += 1

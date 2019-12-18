import h5py
import time
import numpy as np
from sklearn.externals import joblib
import sys

infn = sys.argv[1]
outfn = sys.argv[2]
N = 10201  # length of target data
n = 1000  # current line
part_size = 1000  # samples per part
part = 0  # current part
data = np.empty((part_size, N), dtype=float)
start = time.time()
print('Loading PCA model...')
pca = joblib.load('PCA_target_1500')
print('Loading PCA complete. Time used: ', time.time() - start)
start = time.time()
scaler = joblib.load('Scaler_target')
with h5py.File(outfn, 'w') as opt:
    opt.create_group('/target')
    while n <= 9000:  # loading data
        print(f'Loading data part {part}...')
        start = time.time()
        with h5py.File(infn, 'r') as ipt:
            for i in range(n-1000, n):
                sample = ipt[f'{i:04d}']['isoE'][...]
                data[i-part * 1000] = sample
        print('Loading complete. Time used: ', time.time() - start)
        print(f'Scaling data part {part}...')  # scale input data
        start = time.time()
        data_scaled = scaler.transform(data)
        print('Scaling complete. Time used: ', time.time() - start)
        print(f'Feature extracting with PCA part {part}...')
        start = time.time()
        # data_reduced = pca.transform(data_scaled)
        print('Feature extracting complete. Time used: ', time.time() - start)
        print(f'Saving reduced data part {part}...')
        start = time.time()
        opt['target'][f'{part}'] = data_scaled
        print(f'Complete saving. Time used: ', time.time() - start)
        n += part_size
        part += 1

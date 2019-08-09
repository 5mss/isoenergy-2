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
data = np.empty((part_size, 1), dtype=float)
start = time.time()
scaler = joblib.load('Scaler_sum')
with h5py.File('train_target_sum.h5', 'w') as opt:
    opt.create_group('/target')
    while n <= 9000:
        print(f'Loading data part {part}...')
        start = time.time()
        with h5py.File('train.h5', 'r') as ipt:
            for i in range(n-1000, n):
                sample = ipt[f'{i:04d}']['isoE'][...]
                data[i-part * 1000] = np.sum(np.sum(sample))
        print('Loading complete. Time used: ', time.time() - start)
        print(f'Scaling data part {part}...')
        start = time.time()
        data_scaled = scaler.transform(data)
        print('Scaling complete. Time used: ', time.time() - start)
        print(f'Saving sumed data part {part}...')
        start = time.time()
        data_scaled = data_scaled.reshape(1000,)
        opt['target'][f'{part}'] = data_scaled
        print(f'Complete saving. Time used: ', time.time() - start)
        n += 1000
        part += 1
        print(opt['target'].keys())
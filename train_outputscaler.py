import h5py
import time
import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.externals import joblib
import sys

fn = sys.argv[1]
N = 10201  # length of target data
n = 1000  # current line
nComp = 1000  # n_components in PCA
part_size = 1000  # samples per part
part = 0  # current part
data = np.empty((part_size, N), dtype=float)
start = time.time()
scaler = preprocessing.StandardScaler(copy=False)  # initializing scaler
while n <= 9000:  # training scaler
    print(f'Loading data part {part}...')
    start = time.time()
    with h5py.File(fn, 'r') as ipt:
        for i in range(n-1000, n):
            sample = ipt[f'{i:04d}']['isoE'][...]
            data[i-part * 1000] = sample
    print('Loading complete. Time used: ', time.time() - start)
    print(f'Scaling data part {part}...')
    start = time.time()
    scaler.partial_fit(data)
    print('Scaling complete. Time used: ', time.time() - start)
    n += 1000
    part += 1
joblib.dump(scaler, 'Scaler_target')  # saving scaler

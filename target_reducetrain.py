import h5py
import time
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.externals import joblib
N = 201  # length of target data
N2 = N * N  # N square
n = 9000  # samples per part
nComp = 400  # n_components in PCA
part = 0  # current part
start = time.time()
data = np.empty((n, N2), dtype=float)
print(f'Loading data part {part} from training set...')
with h5py.File('train.h5', 'r') as ipt:
    for i in range(n):
        sample = ipt[f'{i:04d}']['isoE'][...]
        sample = sample.reshape((1, N2))
        data[i] = sample
print('Loading complete. Time used: ', time.time() - start)
start = time.time()
print(f'Scaling data part {part}...')
scaler = joblib.load('Scaler_target')
data_scaled = scaler.transform(data)
print('Scaling complete. Time used: ', time.time() - start)
start = time.time()
print(f'Feature extracting with PCA part {part}...')
pca = decomposition.PCA(n_components=nComp, whiten=False, svd_solver='auto')
pca.fit(data_scaled)
print('Feature extracting complete. Time used: ', time.time() - start)
start = time.time()
print('Saving PCA model...')
joblib.dump(pca, 'PCA_target_4')
print(pca.explained_variance_ratio_)

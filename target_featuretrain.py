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
start = time.time()
data = np.empty((n, N2), dtype=float)
print(f'Loading data part {part} from training set...')
with h5py.File('train.h5', 'r') as ipt:
    for i in range(n):
        sample = ipt[f'{i:04d}']['QPI'][...]
        sample = sample.reshape((1, N2))
        data[i] = sample
print('Loading complete. Time used: ', time.time() - start)
start = time.time()
print(f'Scaling data part {part}...')
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)
print('Scaling complete. Time used: ', time.time() - start)
start = time.time()
print(f'Feature extracting with PCA part {part}...')
pca = decomposition.PCA(n_components=nComp, whiten=False, svd_solver='auto')
pca.fit(data_scaled)
data_reduced = pca.transform(data_scaled)
print('Feature extracting complete. Time used: ', time.time() - start)
start = time.time()
print('Saving PCA model...')
joblib.dump(pca, 'PCA')
print(pca.explained_variance_ratio_)

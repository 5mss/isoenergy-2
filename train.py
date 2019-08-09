import time
import h5py
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import r2_score
import sklearn.neural_network as sk_nn
from sklearn.linear_model import MultiTaskLasso

part = 0  # current part
print('Initializing model...')
nnModel = MultiTaskLasso()
with h5py.File('train_feature_t.h5', 'r') as fin:
    X_test = fin['feature'][f'{part}'][...]
with h5py.File('train_target.h5', 'r') as tin:
    Y_test = tin['target'][f'{part}'][...]
print(f'Loading feature data...')
start = time.time()
with h5py.File('train_feature_t.h5', 'r') as fin:
    X = X_test
    for i in range(1, 9):
        Xa = fin['feature'][f'{i}'][...]
        X = np.concatenate((X, Xa))
print('Loading complete. Time used: ', time.time() - start)
print(f'Loading target data part {part}...')
start = time.time()
with h5py.File('train_target.h5', 'r') as tin:
    Y = Y_test
    for i in range(1, 9):
        Ya = tin['target'][f'{i}'][...]
        Y = np.concatenate((Y, Ya))
print('Loading complete. Time used: ', time.time() - start)
print(f'Training model part {part}...')
start = time.time()
nnModel.fit(X, Y)
print('Training complete. Time used: ', time.time() - start)
print(f'Saving model part {part}...')
joblib.dump(nnModel, 'lsModel')
print('Saving complete. Time used: ', time.time() - start)
print(f'Scoring model part {part}...')
start = time.time()
sc = nnModel.score(X_test, Y_test)
print('Scoring complete. Time used: ', time.time() - start)
print(f'Model score part {part}: ', sc)
part += 1


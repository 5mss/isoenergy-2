import time
import h5py
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import r2_score
import sklearn.neural_network as sk_nn

part = 0  # current part
print('Initializing model...')
nnModel = sk_nn.MLPRegressor(activation='tanh', solver='adam', hidden_layer_sizes=(5500, ), random_state=1)
with h5py.File('train_feature.h5', 'r') as fin:
    X_test = fin['feature'][f'{part}'][...]
with h5py.File('train_target.h5', 'r') as tin:
    Y_test = tin['target'][f'{part}'][...]
while part < 9:
    print(f'Loading feature data part {part}...')
    start = time.time()
    with h5py.File('train_feature.h5', 'r') as fin:
        X = fin['feature'][f'{part}'][...]
    print('Loading complete. Time used: ', time.time() - start)
    print(f'Loading target data part {part}...')
    start = time.time()
    with h5py.File('train_target.h5', 'r') as tin:
        Y = tin['target'][f'{part}'][...]
    print('Loading complete. Time used: ', time.time() - start)
    print(f'Training model part {part}...')
    start = time.time()
    nnModel.partial_fit(X, Y)
    print('Training complete. Time used: ', time.time() - start)
    print(f'Saving model part {part}...')
    joblib.dump(nnModel, 'nnModel')
    print('Saving complete. Time used: ', time.time() - start)
    print(f'Scoring model part {part}...')
    start = time.time()
    sc = nnModel.score(X_test, Y_test)
    print('Scoring complete. Time used: ', time.time() - start)
    print(f'Model score part {part}: ', sc)
    part += 1


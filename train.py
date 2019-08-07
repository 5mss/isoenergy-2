import time
import h5py
import numpy as np
import sklearn.neural_network as sk_nn

part = 0  # current part
print('Initializing model...')
nnModel = sk_nn.MLPRegressor(activation='tanh', solver='adam', max_iter=100)
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
    print(f'Scoring model part {part}...')
    start = time.time()
    sc = nnModel.score(X, Y)
    print('Scoring complete. Time used: ', time.time() - start)
    print(f'Model score part {part}: ', sc)
    part += 1

import time
import h5py
import numpy as np
from sklearn.externals import joblib
from sklearn.kernel_ridge import KernelRidge

part = 2  # current part
n_features = 1000
n_target = 2000
scaler_target = joblib.load('Scaler_target')  # load scaler of target
pca = joblib.load('PCA_target_20')  # load pca model of target
print('Initializing model...')
nnModel = KernelRidge(kernel='poly', degree=3, coef0=8000)  # initialize KernelRidge model
with h5py.File('train_feature_t.h5', 'r') as fin:  # load test data
    X_test = fin['feature']['0'][...]
with h5py.File('train_target_20.h5', 'r') as tin:
    Y_test = tin['target']['0'][...]
print(f'Loading feature data...')
start = time.time()
with h5py.File('train_feature_t.h5', 'r') as fin:  # load training data
    X = X_test
    for i in range(9):
        Xa = fin['feature'][f'{i}'][...]
        X = np.concatenate((X, Xa))
print('Loading complete. Time used: ', time.time() - start)
print(f'Loading target data ...')
start = time.time()
with h5py.File('train_target_20.h5', 'r') as tin:
    Y = Y_test
    for i in range(9):
        Ya = tin['target'][f'{i}'][...]
        Y = np.concatenate((Y, Ya))
print('Loading complete. Time used: ', time.time() - start)
print(f'Training model ...')
start = time.time()
nnModel.fit(X, Y)  # train model
print('Training complete. Time used: ', time.time() - start)
print(f'Saving model ...')
joblib.dump(nnModel, 'krModel_20')  # save model
print('Saving complete. Time used: ', time.time() - start)
print(f'Scoring model ...')  # score model
start = time.time()
sc = nnModel.score(X_test, Y_test)
Y_pred = nnModel.predict(X_test)
Y_pred = pca.inverse_transform(Y_pred)
Y_pred = scaler_target.inverse_transform(Y_pred)
Y_test = pca.inverse_transform(Y_test)
Y_test = scaler_target.inverse_transform(Y_test)
ds = np.linalg.norm(np.abs(Y_test - Y_pred), axis=1)
ds = np.mean(ds)
print('Scoring complete. Time used: ', time.time() - start)
print(f'Model score : ', sc)
print(f'Model distance : ', ds)
part += 1


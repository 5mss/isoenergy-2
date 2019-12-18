import time
import h5py
import numpy as np
from sklearn.externals import joblib
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso
part = 2  # current part
n_features = 1000
n_target = 1000
pca_target = joblib.load('PCA_target_1500')
scaler_target = joblib.load('Scaler_target')
print('Initializing model...')
# Model = RandomForestRegressor()
# Model = MultiTaskElasticNet()  # initialize KernelRidge model
# Model = BaggingRegressor(base_estimator=KernelRidge(kernel='poly', degree=3, coef0=20000), n_estimators=20, random_state=0)
Model = KernelRidge(kernel='poly', degree=3, coef0=15000)

print(f'Loading feature data...')
start = time.time()
with h5py.File('train_class2.h5', 'r') as fin:  # load training data
    X = fin['feature'][...]
    print('Loading complete. Time used: ', time.time() - start)
    print(f'Loading target data ...')
    start = time.time()
    Y = fin['target'][...]
print('Loading complete. Time used: ', time.time() - start)
# Y = pca_target.inverse_transform(Y)
# Y = scaler_target.inverse_transform(Y)
# print(f'Loading problem data ...')
# start = time.time()
# with h5py.File('pred_class2.h5', 'r') as tin:
#     Y = tin['target']
# print('Loading complete. Time used: ', time.time() - start)

Nt = int(len(X) / 10)
X_test = X[0:Nt]
X = X[Nt:]
Y_test = Y[0:Nt]
Y = Y[Nt:]

print(f'Training model ...')
start = time.time()
Model.fit(X, Y)  # train model
print('Training complete. Time used: ', time.time() - start)
print(f'Saving model ...')
joblib.dump(Model, 'testModel')  # save model
print('Saving complete. Time used: ', time.time() - start)
print(f'Scoring model ...')  # score model
score = Model.score(X_test, Y_test)
start = time.time()
Y_pred = Model.predict(X_test)
# Y_pred = pca.inverse_transform(Y_pred)
# Y_pred = scaler_target.inverse_transform(Y_pred)
# Y_test = pca.inverse_transform(Y_test)
# Y_test = scaler_target.inverse_transform(Y_test)
# ds = np.linalg.norm(np.abs(Y_test - Y_pred), axis=1)
# ds = np.mean(ds)
# print('Scoring complete. Time used: ', time.time() - start)
print(f'Model score : ', score)
# print(f'Model distance : ', ds)
part += 1


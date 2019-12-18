from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sys
import time
import h5py
import numpy as np
from sklearn.externals import joblib

train_fn = sys.argv[1]
problem_fn = sys.argv[2]
part = 1  # current part
n_features = 1000
n_target = 2000
scaler_target = joblib.load('Scaler_target')  # load scaler of target
pca = joblib.load('PCA_target_1500')  # load pca model of target
print('Initializing model...')
Model = KMeans(n_clusters=2)
print(f'Loading feature data...')
start = time.time()
with h5py.File(train_fn, 'r') as fin:  # load training data
    i = 0
    X = fin['feature'][f'{i}'][...]
    for i in range(1, 9):
        Xa = fin['feature'][f'{i}'][...]
        X = np.concatenate((X, Xa))
# with h5py.File(problem_fn, 'r') as fin:  # load training data
#     i = 0
#     X_pred = fin['feature'][f'{i}'][...]
#     for i in range(1, 9):
#         X_preda = fin['feature'][f'{i}'][...]
#         X_pred = np.concatenate((X_pred, X_preda))

# X = np.concatenate((X, X_pred))
print(f'Loading target data ...')
start = time.time()
with h5py.File('output_unreduced.h5', 'r') as tin:
    i = 0
    Y = tin['target'][f'{i}'][...]
    for i in range(1, 9):
        Ya = tin['target'][f'{i}'][...]
        Y = np.concatenate((Y, Ya))
print('Loading complete. Time used: ', time.time() - start)
print('Loading complete. Time used: ', time.time() - start)
print(f'Training model ...')
start = time.time()
Model.fit(X)  # train model
print('Training complete. Time used: ', time.time() - start)
print(f'Saving model ...')
joblib.dump(Model, 'Cluster')  # save model
print('Saving complete. Time used: ', time.time() - start)

labels = Model.predict(X)

with h5py.File('labels.h5', 'w') as opt:
    opt['labels'] = labels
X1 = []
X2 = []
Y1 = []
Y2 = []
for i in range(9000):
    if labels[i] == 0:
        X1.append(X[i])
        Y1.append(Y[i])
    else:
        X2.append(X[i])
        Y2.append(Y[i])
with h5py.File('train_class1.h5', 'w') as c1:
    c1['feature'] = X
    c1['target'] = Y
del X1
del Y1
with h5py.File('train_class2.h5', 'w') as c2:
    c2['feature'] = X2
    c2['target'] = Y2
del X2
del Y2

# X1 = []
# X2 = []
# for i in range(9000, 9900):
#     if labels[i] == 0:
#         X1.append(X[i])
#     else:
#         X2.append(X[i])
#
# with h5py.File('pred_class1.h5', 'w') as c1:
#     c1['feature'] = X1
# del X1
# with h5py.File ('pred_class2.h5', 'w') as c2:
#     c2['feature'] = X2


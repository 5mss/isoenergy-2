import h5py
import matplotlib.pyplot as plt
from sklearn.externals import joblib

with h5py.File('labels_GMM.h5', 'r') as ipt:
    labels = ipt['labels'][...]

pca_target = joblib.load('PCA_target_1500')
scaler_target = joblib.load('Scaler_target')

with h5py.File('train_class1.h5', 'r') as ipt:
    for i in range(40):
        data = ipt[f'{i:04d}']['isoE'][...]
        c = labels[i]
        plt.figure()
        plt.title(f'data {i} -- class {c}')
        plt.imshow(data)
        plt.show()

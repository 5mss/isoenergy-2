import numpy as np
import pandas as pd
import h5py
import sys
import time
from sklearn.externals import joblib

fn = sys.argv[1]
optName = 'train_output_polar.h5'
# optName = 'test.h5'
# pca_target = joblib.load('PCA_target_1500')
# scaler_target = joblib.load('Scaler_target')


start = time.time()
with h5py.File(fn, 'r') as ipt:
    keys = list(ipt.keys())
    dsetName = list(ipt[keys[0]].keys())[1]  # get group name
    dataShape = ipt[keys[0]][dsetName].shape
    H = 101
    W = 101
    pixels = H * W
    x = np.linspace(1, W, W)
    y = np.linspace(1, H, H)
    xm, ym = np.meshgrid(x, y)
    tangentM = ym / xm
    tangentL = tangentM.reshape((pixels,))
    radiusM = np.sqrt(xm * xm + ym * ym)
    radiusL = tangentL.reshape((pixels,))
    dFrame = pd.DataFrame({'tangent': tangentL, 'radius': radiusL, 'index': np.arange(0, pixels)})
    dFrame = dFrame.sort_values(by=['tangent', 'radius'], axis=0, ascending=[True, True])
    seq = np.array(dFrame['index'])
    with h5py.File(optName, 'w') as opt:
        count = 0
        for k in keys:
        # for i in range(20):
            data = np.array(ipt[k][dsetName])
            # data = ipt['target'][...][i]
            # data = pca_target.inverse_transformerse_transform(data)
            # data = scaler_target.inverse_transform(data)
            data = data[W-1:, H-1:]
            data = data.reshape((pixels,))
            dataorig = np.empty(pixels)
            for j in range(pixels):
                dataorig[j] = data[seq[j]]
            # dataorig = dataorig.reshape(101, 101)
            # data1 = dataorig[100::-1, :]
            # data3 = dataorig[:, 100::-1]
            # data2 = dataorig[100::-1, 100::-1]
            # dataorig = np.concatenate((np.concatenate((data2, data3), axis=0), np.concatenate((data1, dataorig), axis=0)), axis=1)
            opt.create_group(k)
            # opt.create_group(f'{i:04d}')
            opt[k][dsetName] = dataorig
            print('Data ' + k + f' done processing. Time used {time.time() - start}')
            start = time.time()
        print(f'Number of data processed: {len(list(opt.keys()))}')

print(f'Time used: {time.time() - start}')



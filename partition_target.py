import h5py
import time
import numpy as np

# partition training isoE data in 9 parts

N = 201  # length of target data
N2 = N*N
n = 1000  # current line
part_size = 1000  # samples per part
part = 0  # current part
data = np.empty((part_size, N2), dtype=float)
with h5py.File('train_target_unreduced.h5', 'w') as opt:
    opt.create_group('/target')
    while n <= 9000:
        with h5py.File('train.h5', 'r') as ipt:
            for i in range(n - 1000, n):
                sample = ipt[f'{i:04d}']['isoE'][...]
                data[i - part * 1000] = sample.reshape((1, N2))
        opt['target'][f'{part}'] = data
        part += 1
        n += 1000
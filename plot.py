import h5py
import matplotlib.pyplot as plt

with h5py.File('test.h5', 'r') as ipt:
    for i in range(20):
        data = ipt[f'{i:04d}']['isoE'][...]
        plt.imshow(data)
        plt.show()

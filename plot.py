import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('answer_sum.h5', 'r') as ipt:
    data = ipt['0001']['isoE'][...]
plt.imshow(data)
plt.show()

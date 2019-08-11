import h5py
import matplotlib.pyplot as plt

with h5py.File('answer_kr_20.h5', 'r') as ipt:
    data = ipt['0001']['isoE'][...]
plt.imshow(data)
plt.show()

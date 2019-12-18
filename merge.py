import numpy as np
import h5py
import sys

fn1 = sys.argv[1]
fn2 = sys.argv[2]

with h5py.File('labels.h5', 'r') as ipt:
    labels = ipt['labels'][...]

labels = labels[9000:]
print(f'number of class 2 in labels: {sum(labels)}')

i = 0
j = 0

ipt1 = h5py.File(fn1, 'r')
ipt2 = h5py.File(fn2, 'r')
opt = h5py.File('answer_polar.h5', 'w')

print(f'number of class 1 in input: {len(list(ipt1.keys()))}')
print(f'number of class 2 in input: {len(list(ipt2.keys()))}')

for k, t in enumerate(labels):
    opt.create_group(f'{k:04d}')
    if t == 0:
        opt[f'{k:04d}']['isoE'] = ipt1[f'{i:04d}']['isoE'][...]
        i += 1
    else:
        opt[f'{k:04d}']['isoE'] = ipt1[f'{j:04d}']['isoE'][...]
        j += 1

ipt1.close()
ipt2.close()
opt.close()


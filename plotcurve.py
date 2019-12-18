import matplotlib.pyplot as plt
import numpy as np
import joblib

model = joblib.load('krModel_20')
data = np.arange(1000, 10000, 1000)

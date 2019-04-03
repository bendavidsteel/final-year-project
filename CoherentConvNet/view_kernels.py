import numpy as np 
import matplotlib.pyplot as plt
import pickle

with open('', 'rb') as f:
    data = pickle.load(f)

[params, cost, cost_val, num_epochs] = data

[f1, f2, w3, w4, b1, b2, b3, b4] = params

(_, n_f, f, f) = f1.shape

fig, axes = plt.subplots(nrows=1, ncols=n_f)

for i in range(n_f):
    axes[i].quiver(np.real(f1[0,i,:,:]), np.imag(f1[0,i,:,:]))

plt.show()
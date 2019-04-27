'''Author: Ben Steel
Date: 26/03/19'''

import numpy as np 
import matplotlib.pyplot as plt
import pickle

save_path = '4Gamma_lr01_bias2b_CoherentNet_4_8f5_8p2_128_fourshapesDataset_NLdata_try0.pkl'

with open(save_path, 'rb') as f:
    data = pickle.load(f)

[params, cost, cost_val, nl1_p, nl2_p, nl3_p, nl4_p, final_layer] = data

[f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params

(n_f, _, f, f) = f1.shape

fig, axes = plt.subplots(nrows=1, ncols=n_f)

for i in range(n_f):
    axes[i].quiver(np.real(f1[i,0,:,:]), np.imag(f1[i,0,:,:]), angles='xy', scale=1, width=0.01)
    axes[i].set_aspect('equal')
    axes[i].set_title('Kernel ' + str(i + 1))

    for tic in axes[i].xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

    for tic in axes[i].yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

plt.show()
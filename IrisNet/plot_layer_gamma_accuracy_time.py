import matplotlib.pyplot as plt
import numpy as np 
import pickle

if __name__ == '__main__':

    eval_save_path = "layer_gamma_accuracy_1632_0.05_15.pkl"

    num_gammas = 20
    iters = 5

    to_save = pickle.load(open(eval_save_path, 'rb')) 

    [x_g, y_g, g_a, e_n] = to_save

    x = y_g[:num_gammas]
    y = x

    z = np.zeros((num_gammas, num_gammas))

    for i in range(num_gammas):
        for j in range(num_gammas):
            z[j,i] = g_a[i*num_gammas + j]

    plt.contourf(x, y, z, 20)
    plt.xlabel(r'$\kappa$ for First Layer')
    plt.ylabel(r'$\kappa$ for Second Layer')
    cbar = plt.colorbar()
    # cbar.set_label("Num Epochs to Best Performance")
    cbar.set_label("Test Set Accuracy")
    plt.show()
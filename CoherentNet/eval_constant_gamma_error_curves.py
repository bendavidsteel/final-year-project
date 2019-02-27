''' Author: Ben Steel '''


from NN.network_simple_dataset import *
from NN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':

    gamma_error_curves = {}

    for gamma in np.linspace(0.001, 1, 10):
        
        cost = train(save = False, gamma = gamma)
        gamma_error_curves[gamma] = cost
    
    save_path = "gamma_error_curves_0.001t10t1_simpledataset_singlelayer128"

    with open(save_path, 'wb') as file:
        pickle.dump(gamma_error_curves, file)

    gamma_error_curves = pickle.load(open(save_path, 'rb'))

    # Plot cost 

    legend = []

    for gamma in gamma_error_curves:
        if gamma == 0.001:
            continue
        plt.plot(gamma_error_curves[gamma])
        legend.append("Gamma = " + str(gamma))

    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend(legend, loc='upper right')
    plt.show()
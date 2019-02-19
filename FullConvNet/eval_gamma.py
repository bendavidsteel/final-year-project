''' Author: Ben Steel '''


from CNN.full_network_batch_simple_dataset import *
from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':

    gamma_error_curves = {}

    for gamma in np.linspace(0.1, 1, 10):
        
        cost = train(save = False, gamma = gamma)
        gamma_error_curves[gamma] = cost
    
    save_path = "gamma_error_curves_0.1t10t1_simpledataset_full"

    with open(save_path, 'wb') as file:
        pickle.dump(gamma_error_curves, file)

    gamma_error_curves = pickle.load(open(save_path, 'rb'))

    # Plot cost 

    legend = []

    for gamma in gamma_error_curves:
        plt.plot(gamma_error_curves[gamma])
        legend.append("Gamma = " + str(gamma))

    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend(legend, loc='upper right')
    plt.show()
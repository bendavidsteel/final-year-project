''' Author: Ben Steel
    Created: 17/03/19'''

import matplotlib.pyplot as plt
import numpy as np
import pickle


if __name__ == '__main__':

    gammas = np.linspace(0.1, 3, 4)

    # Plot cost 

    factor = 100
    offset = 5

    fig, ax = plt.subplots(nrows=1, ncols=1)

    save_path = "gamma_error_curves_" + str(gammas[0]) + "_heartdisease_coherentnet1616"
    saved = pickle.load(open(save_path, 'rb'))

    q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    interval = max_len // factor

    error = [q75_cost[::interval] - q50_cost[::interval],q50_cost[::interval] - q25_cost[::interval]]
    ax.errorbar([x for x in range(0, max_len, interval)], q50_cost[::interval], yerr=error, color='b')

    save_path = "gamma_error_curves_" + str(gammas[1]) + "_heartdisease_coherentnet1616"
    saved = pickle.load(open(save_path, 'rb'))

    q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    error = [q75_cost[::interval] - q50_cost[::interval],q50_cost[::interval] - q25_cost[::interval]]
    ax.errorbar([x for x in range(offset, max_len, interval)], q50_cost[::interval], yerr=error, color='g')

    save_path = "gamma_error_curves_" + str(gammas[2]) + "_heartdisease_coherentnet1616"
    saved = pickle.load(open(save_path, 'rb'))

    q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    error = [q75_cost[::interval] - q50_cost[::interval],q50_cost[::interval] - q25_cost[::interval]]
    ax.errorbar([x for x in range(offset*2, max_len, interval)], q50_cost[::interval], yerr=error, color='r')

    save_path = "gamma_error_curves_" + str(gammas[3]) + "_heartdisease_coherentnet1616"
    saved = pickle.load(open(save_path, 'rb'))

    q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    error = [q75_cost[::interval] - q50_cost[::interval],q50_cost[::interval] - q25_cost[::interval]]
    ax.errorbar([x for x in range(offset*3, max_len, interval)], q50_cost[::interval], yerr=error, color='c')

    ax.set_xlabel('# Iterations')
    ax.set_ylabel('Cost')
    ax.legend(["Training", "Lowest Validation Cost"], loc='upper right')

    plt.show()
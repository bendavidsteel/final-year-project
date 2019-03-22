''' Author: Ben Steel
    Created: 17/03/19'''

import matplotlib.pyplot as plt
import numpy as np
import pickle


if __name__ == '__main__':

    gammas = np.linspace(0.1, 3, 5)
    colors = ['b', 'g', 'r', 'c', 'm']

    # Plot cost 

    factor = 200
    offset = 10
    legend = []

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for i in range(5):
        save_path = "gamma_error_curves_" + str(gammas[i]) + "_heartdisease_coherentnet1616"
        saved = pickle.load(open(save_path, 'rb'))

        legend.append(r'$\kappa$ = ' + str(gammas[i]))

        q25_cost, q50_cost, q75_cost, min_len, max_len = saved

        interval = max_len // factor

        ax.plot([min_len, max_len], [0.3 - 0.01*i, 0.3 - 0.01*i], color=colors[i], marker='|')

        error = [q75_cost[::interval] - q50_cost[::interval],q50_cost[::interval] - q25_cost[::interval]]
        ax.errorbar([x for x in range(offset*i, max_len + offset*i, interval)], q50_cost[::interval], yerr=error, color=colors[i], errorevery = 10)

    # save_path = "gamma_error_curves_" + str(gammas[1]) + "_heartdisease_coherentnet1616"
    # saved = pickle.load(open(save_path, 'rb'))

    # legend.append(r'$\kappa$ = ' + str(gammas[1]))

    # q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    # error = [q75_cost[::interval] - q50_cost[::interval],q50_cost[::interval] - q25_cost[::interval]]
    # ax.errorbar([x for x in range(offset, max_len + offset, interval)], q50_cost[::interval], yerr=error, color='g')

    # save_path = "gamma_error_curves_" + str(gammas[2]) + "_heartdisease_coherentnet1616"
    # saved = pickle.load(open(save_path, 'rb'))

    # legend.append(r'$\kappa$ = ' + str(gammas[2]))

    # q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    # error = [q75_cost[::interval] - q50_cost[::interval],q50_cost[::interval] - q25_cost[::interval]]
    # ax.errorbar([x for x in range(offset*2, max_len + offset*2, interval)], q50_cost[::interval], yerr=error, color='r')

    # save_path = "gamma_error_curves_" + str(gammas[3]) + "_heartdisease_coherentnet1616"
    # saved = pickle.load(open(save_path, 'rb'))

    # legend.append(r'$\kappa$ = ' + str(gammas[3]))

    # q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    # error = [q75_cost[::interval] - q50_cost[::interval],q50_cost[::interval] - q25_cost[::interval]]
    # ax.errorbar([x for x in range(offset*3, max_len + offset*3, interval)], q50_cost[::interval], yerr=error, color='c')

    ax.set_xlabel('# Iterations')
    ax.set_ylabel('Cost')
    ax.legend(legend, loc='upper right')

    plt.show()
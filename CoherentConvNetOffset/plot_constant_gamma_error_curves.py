''' Author: Ben Steel
    Created: 17/03/19'''

import matplotlib.pyplot as plt
import numpy as np
import pickle


if __name__ == '__main__':

    gammas = np.linspace(0.1, 3, 4)

    save_path = "gamma_error_curves_" + str(gammas[0]) + "_heartdisease_coherentnet1616"
    
    saved = pickle.load(open(save_path, 'rb'))

    # Plot cost 

    interval = 100

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    pre_error = [q75_cost[:min_len:interval] - q50_cost[:min_len:interval],q50_cost[:min_len:interval] - q25_cost[:min_len:interval]]
    post_error = [q75_cost[min_len::interval] - q50_cost[min_len::interval],q50_cost[min_len::interval] - q25_cost[min_len::interval]]

    ax1.errorbar([x for x in range(0, min_len, interval)], q50_cost[:min_len:interval], yerr=pre_error)
    ax1.errorbar([x for x in range(min_len, max_len, interval)], q50_cost[min_len::interval], yerr=post_error)

    ax1.set_xlabel('# Iterations')
    ax1.set_ylabel('Cost')
    ax1.legend(["Training", "Lowest Validation Cost"], loc='upper right')
    ax1.set_title("Gamma = " + str(gammas[0]))

    save_path = "gamma_error_curves_" + str(gammas[1]) + "_heartdisease_coherentnet1616"
    
    saved = pickle.load(open(save_path, 'rb'))

    q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    pre_error = [q75_cost[:min_len:interval] - q50_cost[:min_len:interval],q50_cost[:min_len:interval] - q25_cost[:min_len:interval]]
    post_error = [q75_cost[min_len::interval] - q50_cost[min_len::interval],q50_cost[min_len::interval] - q25_cost[min_len::interval]]

    ax2.errorbar([x for x in range(0, min_len, interval)], q50_cost[:min_len:interval], yerr=pre_error)
    ax2.errorbar([x for x in range(min_len, max_len, interval)], q50_cost[min_len::interval], yerr=post_error)

    ax2.set_xlabel('# Iterations')
    ax2.set_ylabel('Cost')
    ax2.legend(["Training", "Lowest Validation Cost"], loc='upper right')
    ax2.set_title("Gamma = " + str(gammas[1]))

    save_path = "gamma_error_curves_" + str(gammas[2]) + "_heartdisease_coherentnet1616"
    
    saved = pickle.load(open(save_path, 'rb'))

    q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    pre_error = [q75_cost[:min_len:interval] - q50_cost[:min_len:interval],q50_cost[:min_len:interval] - q25_cost[:min_len:interval]]
    post_error = [q75_cost[min_len::interval] - q50_cost[min_len::interval],q50_cost[min_len::interval] - q25_cost[min_len::interval]]

    ax3.errorbar([x for x in range(0, min_len, interval)], q50_cost[:min_len:interval], yerr=pre_error)
    ax3.errorbar([x for x in range(min_len, max_len, interval)], q50_cost[min_len::interval], yerr=post_error)

    ax3.set_xlabel('# Iterations')
    ax3.set_ylabel('Cost')
    ax3.legend(["Training", "Lowest Validation Cost"], loc='upper right')
    ax3.set_title("Gamma = " + str(gammas[2]))

    save_path = "gamma_error_curves_" + str(gammas[3]) + "_heartdisease_coherentnet1616"
    
    saved = pickle.load(open(save_path, 'rb'))

    q25_cost, q50_cost, q75_cost, min_len, max_len = saved

    pre_error = [q75_cost[:min_len:interval] - q50_cost[:min_len:interval],q50_cost[:min_len:interval] - q25_cost[:min_len:interval]]
    post_error = [q75_cost[min_len::interval] - q50_cost[min_len::interval],q50_cost[min_len::interval] - q25_cost[min_len::interval]]

    ax4.errorbar([x for x in range(0, min_len, interval)], q50_cost[:min_len:interval], yerr=pre_error)
    ax4.errorbar([x for x in range(min_len, max_len, interval)], q50_cost[min_len::interval], yerr=post_error)

    ax4.set_xlabel('# Iterations')
    ax4.set_ylabel('Cost')
    ax4.legend(["Training", "Lowest Validation Cost"], loc='upper right')
    ax4.set_title("Gamma = " + str(gammas[3]))

    plt.tight_layout()
    plt.show()
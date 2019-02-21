import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_2overpiGamma_Net128_SimpleDigits_NLdeltadata'

    params, cost, final_nl, final_d = pickle.load(open(save_path, 'rb'))

    [nl1, nl2, nl3] = final_nl
    [d1, d2, d3] = final_d

    fig, axes = plt.subplots(3,1)

    axes[0].scatter(d1, nl1)

    axes[1].scatter(d2, nl2)

    axes[2].scatter(d3, nl3)

    plt.xlabel('Delta')
    plt.ylabel('Activation Value')
    plt.show()
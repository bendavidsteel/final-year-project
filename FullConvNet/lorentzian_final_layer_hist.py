import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_2overpiGamma_FullNet6464_SimpleDigits_NLquartiledata'

    params, cost, layer_q5, layer_q25, layer_q50, layer_q75, layer_q95, final_layer = pickle.load(open(save_path, 'rb'))

    [nl1, nl2, nl3, nl4, nl5] = final_layer

    legend = ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5"]

    nl1_h = np.histogram(nl1.flatten(), density=True)
    nl2_h = np.histogram(nl2.flatten(), density=True)
    nl3_h = np.histogram(nl3.flatten(), density=True)
    nl4_h = np.histogram(nl4.flatten(), density=True)
    nl5_h = np.histogram(nl5.flatten(), density=True)

    x1 = [(nl1_h[1][i] + nl1_h[1][i+1])/2 for i in range(len(nl1_h[1]) - 1)]
    plt.plot(x1, nl1_h[0])

    x2 = [(nl2_h[1][i] + nl2_h[1][i+1])/2 for i in range(len(nl2_h[1]) - 1)]
    plt.plot(x2, nl2_h[0])

    x3 = [(nl3_h[1][i] + nl3_h[1][i+1])/2 for i in range(len(nl3_h[1]) - 1)]
    plt.plot(x3, nl3_h[0])

    x4 = [(nl4_h[1][i] + nl4_h[1][i+1])/2 for i in range(len(nl4_h[1]) - 1)]
    plt.plot(x4, nl4_h[0])

    x5 = [(nl5_h[1][i] + nl5_h[1][i+1])/2 for i in range(len(nl5_h[1]) - 1)]
    plt.plot(x5, nl5_h[0])

    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.legend(legend, loc='upper right')
    plt.show()
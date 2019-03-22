from NN.network import *
from NN.utils import *

import matplotlib.pyplot as plt
import numpy as np 
import pickle

if __name__ == '__main__':

    iters = 10
    num_gammas = 9

    g_vals = np.linspace(0.1, 1, num_gammas)

    x_g = np.zeros((num_gammas**3,))
    y_g = np.zeros((num_gammas**3,))
    # z_a = np.zeros((num_gammas**3,))
    g_a = np.zeros((num_gammas**3,))
    e_n = np.zeros((num_gammas**3,))


    for i in range(num_gammas):
        for j in range(num_gammas):
            # for k in range(num_gammas):
            for n in range(iters):

                # save_path = 'layer_gamma_accuracy/_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
                save_path = 'layer_gamma_accuracy/_'+str(i)+'_'+str(j)+'.pkl'
                # cost = train(gamma = [g_vals[i], g_vals[j], g_vals[k]], layers = [32,32], save_path = save_path)
                cost = train(gamma = [g_vals[i], g_vals[j]], layers = [32,32], save_path = save_path)

                params, cost, cost_val, num_epochs = pickle.load(open(save_path, 'rb'))

                # Get test data
                X, y_dash = heart_testing_set()
                # Normalize the data
                test_data = norm_stack_shuffle(X,y_dash)

                num_classes = 2
                
                X = test_data[:,0:-1]
                y = test_data[:,-1]

                corr = 0

                for k in range(len(X)):
                    x = X[k]
                    # pred, prob = predict(x, y, params, [g_vals[i], g_vals[j], g_vals[k]])
                    pred, prob = predict(x, y, params, [g_vals[i], g_vals[j]])
                    if pred==y[k]:
                        corr+=1
                    
                # x_g[(i*(num_gammas**2)) + (j*num_gammas) + k] = g_vals[i]
                # y_g[(i*(num_gammas**2)) + (j*num_gammas) + k] = g_vals[j]
                # z_g[(i*(num_gammas**2)) + (j*num_gammas) + k] = g_vals[k]
                # g_a[(i*(num_gammas**2)) + (j*num_gammas) + k] += float(corr/len(test_data)*100))
                # e_n[(i*(num_gammas**2)) + (j*num_gammas) + k] += num_epochs

                x_g[i*num_gammas + j] = g_vals[i]
                y_g[i*num_gammas + j] = g_vals[j]
                g_a[i*num_gammas + j] += float(corr/len(test_data)*100)
                e_n[i*num_gammas + j] += num_epochs

                # t.set_description("x: %.2f, y: %.2f, z: %.2f, n: %.2f" % (g_vals[i], g_vals[j], g_vals[k], n))
                print("x: %.2f, y: %.2f, n: %.2f" % (g_vals[i], g_vals[j], n))

    # taking the average
    g_a /= iters
    e_n /= iters

    # to_save = [x_g, y_g, z_g, g_a, e_n]
    to_save = [x_g, y_g, g_a, e_n]
    save_path = "layer_gamma_accuracy_1616.pkl"

    to_save = pickle.load(open(save_path, 'rb'))
        
    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)

    plt.scatter(x_a, y_a, s = g_a, c=g_a)
    plt.xlabel("gamma for first layer")
    plt.ylabel("gamma for second layer")
    plt.show()
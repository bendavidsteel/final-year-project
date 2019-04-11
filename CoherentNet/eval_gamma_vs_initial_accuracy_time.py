from NN.network_alt import *
from NN.utils import *

import matplotlib.pyplot as plt
import numpy as np 
import pickle

if __name__ == '__main__':

    eval_save_path = "gamma_vs_init_accuracy_coherent_3232_heart_ne5000.pkl"

    num_gammas = 15
    num_inits = 15

    iters = 3

    gamma_vals = np.linspace(0.05, 2, num_gammas)
    init_vals = np.linspace(0, 4, num_inits)

    gamma = np.zeros((num_gammas*num_inits,))
    inits = np.zeros((num_gammas*num_inits,))
    acc = np.zeros((num_gammas*num_inits,))
    epochs = np.zeros((num_gammas*num_inits,))

    # to_save = pickle.load(open(eval_save_path, 'rb')) 

    # [x_g, y_g, g_a, e_n] = to_save

    for i in range(num_gammas):
        for j in range(num_inits):
            # for k in range(num_gammas):

            acc[i*num_gammas + j] = 0
            epochs[i*num_gammas + j] = 0

            for n in range(iters):

                # save_path = 'layer_gamma_accuracy/_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
                save_path = 'gamma_vs_init_accuracy/_'+str(i)+'_'+str(j)+'.pkl'
                # cost = train(gamma = [g_vals[i], g_vals[j], g_vals[k]], layers = [32,32], save_path = save_path)
                cost = train(gamma = gamma_vals[i], bias_init = init_vals[j], layers = [32, 32], save_path = save_path, progress_bar=False, lr=0.01, max_epochs=5000)

                params, cost, cost_val, num_epochs = pickle.load(open(save_path, 'rb'))

                # Get test data
                # X, y_dash = iris_testing_set()
                X, y_dash = heart_testing_set()
                # Normalize the data
                test_data = norm_stack_shuffle(X,y_dash)

                # num_classes = 3
                num_classes = 2
                
                X = test_data[:,0:-1]
                y = test_data[:,-1]

                corr = 0

                for k in range(len(X)):
                    x = X[k]
                    # pred, prob = predict(x, y, params, [g_vals[i], g_vals[j], g_vals[k]])
                    pred, prob = predict(x, y, params, gamma_vals[i])
                    if pred==y[k]:
                        corr+=1
                    
                # x_g[(i*(num_gammas**2)) + (j*num_gammas) + k] = g_vals[i]
                # y_g[(i*(num_gammas**2)) + (j*num_gammas) + k] = g_vals[j]
                # z_g[(i*(num_gammas**2)) + (j*num_gammas) + k] = g_vals[k]
                # g_a[(i*(num_gammas**2)) + (j*num_gammas) + k] += float(corr/len(test_data)*100))
                # e_n[(i*(num_gammas**2)) + (j*num_gammas) + k] += num_epochs

                gamma[i*num_gammas + j] = gamma_vals[i]
                inits[i*num_gammas + j] = init_vals[j]
                acc[i*num_gammas + j] += float(corr/len(test_data)*100) / iters
                epochs[i*num_gammas + j] += num_epochs / iters

                # t.set_description("x: %.2f, y: %.2f, z: %.2f, n: %.2f" % (g_vals[i], g_vals[j], g_vals[k], n))
                print("g: %.2f, i: %.2f, n: %.2f" % (gamma_vals[i], init_vals[j], n))

        to_save = [gamma, inits, acc, epochs]

        with open(eval_save_path, 'wb') as file:
            pickle.dump(to_save, file)

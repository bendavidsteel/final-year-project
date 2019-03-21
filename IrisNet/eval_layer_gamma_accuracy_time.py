from NN.network_heart import *
from NN.utils import *

import matplotlib.pyplot as plt
import numpy as np 
import pickle

if __name__ == '__main__':

    eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_bias5b.pkl"

    num_gammas = 10

    iters = 5

    g_vals = np.linspace(0.05, 15, num_gammas)

    x_g = np.zeros((num_gammas**2,))
    y_g = np.zeros((num_gammas**2,))
    # z_a = np.zeros((num_gammas**3,))
    g_a = np.zeros((num_gammas**2,))
    e_n = np.zeros((num_gammas**2,))

    # to_save = pickle.load(open(eval_save_path, 'rb')) 

    # [x_g, y_g, g_a, e_n] = to_save

    for i in range(num_gammas):
        for j in range(num_gammas):
            # for k in range(num_gammas):

            g_a[i*num_gammas + j] = 0
            e_n[i*num_gammas + j] = 0

            for n in range(iters):

                # save_path = 'layer_gamma_accuracy/_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
                save_path = 'layer_gamma_accuracy/_'+str(i)+'_'+str(j)+'.pkl'
                # cost = train(gamma = [g_vals[i], g_vals[j], g_vals[k]], layers = [32,32], save_path = save_path)
                cost = train(gamma = [g_vals[i], g_vals[j]], layers = [32, 32], save_path = save_path, progress_bar=False, lr=0.01, max_epochs=5000)

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
                g_a[i*num_gammas + j] += float(corr/len(test_data)*100) / iters
                e_n[i*num_gammas + j] += num_epochs / iters

                # t.set_description("x: %.2f, y: %.2f, z: %.2f, n: %.2f" % (g_vals[i], g_vals[j], g_vals[k], n))
                print("x: %.2f, y: %.2f, n: %.2f" % (g_vals[i], g_vals[j], n))

        to_save = [x_g, y_g, g_a, e_n]

        with open(eval_save_path, 'wb') as file:
            pickle.dump(to_save, file)

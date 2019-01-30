#Author: Ben Steel Date: 29/01/19

import numpy as np
import BetterNeuralNetworks as bnn
import matplotlib.pyplot as plt

def evalLorentzParameters():
    x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    layers = [4]

    alpha = 0.001

    iters = 10

    gammas = np.linspace(-2, 2, 10)
    x0s = np.linspace(-2, 2, 10)
    errors = np.zeros((10,10))
    
    file = open('ActivationLorentzx0GammaErrorXOR.txt', 'w')
    file.write("Gamma ; x0 ; Error after 10000 iterations, averaged over 10 times, single hidden layer with 4 nodes #\n")

    for i in range(len(gammas)):
        # find average performance
        for j in range(len(x0s)):
            for k in range(iters):

                model = bnn.ActivationLorentzianNeuralNetwork(x, y, layers, alpha = alpha, gamma = gammas[i], x0 = x0s[j])

                model.train(10000)

                errors[i][j] += np.sum(np.square(model.predict(x) - y))

            errors[i][j] /= iters

            file.write(str(gammas[i]) + ' ; ' + str(x0s[j]) + ' ; ' + str(errors[i][j]) + '#' + '\n')

            print(str((10*i) + j + 1) + "%")

    file.close()

    plt.contour(x0s, gammas, errors)
    plt.ylabel("Gamma")
    plt.xlabel("x0")
    plt.title("Final error for different values of Gamma and x0")
    plt.show()


evalLorentzParameters()
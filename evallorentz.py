# written by ben steel
# 24/10/18
# cma library taken from https://github.com/CMA-ES/pycma

import neuralnetwork as nn
import BetterNeuralNetworks as bnn
import PyTorchNeuralNetworks as pnn
import numpy as np
from sklearn.neural_network import MLPRegressor
# import cma
import matplotlib.pyplot as plt
import time

def evalLorentzErrorCurve(alpha, gamma, layers):
    x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    model = bnn.ActivationLorentzianNeuralNetwork(x, y, layers, alpha = alpha, gamma = gamma, x0 = -1)
    
    training_iters = 1000

    m = np.zeros(training_iters)
    n = np.zeros(training_iters)

    print(model.predict(x))

    for i in range(training_iters):
        model.train(10)

        m[i] = i * 10
        n[i] = np.sum(np.square(model.predict(x) - y))

    print(model.predict(x))

    plt.plot(m, n)
    plt.axis([0, np.max(m), 0, np.max(n)])
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.title("Error decreasing with increasing training iterations, with Gamma = " + str(gamma))
    plt.show()

# es = cma.CMAEvolutionStrategy(9*[0], 0.5)
# es.optimize(evalLorentz)

# print(es.result_pretty())

def evalLorentzGamma():
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

    m = np.linspace(-2, 2, 100)
    n = np.zeros(100)
    
    file = open('FullDerivLorentzGammaErrorXOR.txt', 'w')
    file.write("Gamma ; Error after 10000 iterations, 10 iterations ; first layer weights ; output layer weights\n")

    for i in range(len(m)):
        # find average performance
        for j in range(iters):

            model = bnn.FullDerivLorentzianNeuralNetwork(x, y, layers, alpha, m[i])

            model.train(10000)

            n[i] += np.sum(np.square(model.predict(x) - y))

        n[i] /= iters

        file.write(str(m[i]) + ' ; ' + str(n[i]) + ' ; ' + str(model.weights_x0) + '\n')

        print(str(i) + "%")

    file.close()

    best_gamma = m[np.argmin(n)]

    print("Value of gamma with lowest error is " + str(best_gamma))

    plt.plot(m, n)
    plt.axis([np.min(m), np.max(m), 0, np.max(n)])
    plt.ylabel("Final Error")
    plt.xlabel("Gamma")
    plt.title("Final error for different values of Gamma")
    plt.show()

def evalLorentzComplexity():
    x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    alpha = 0.001

    gamma = 1

    iters = 10

    error = np.zeros((19,19))

    file = open('SqrtLorentzComplexityErrorXOR.txt', 'w')
    file.write("Layer Size, No Layers; Error after 10000 iterations with Gamma 1\n")

    for layer_size in range(2, 21):
        print(layer_size)
        for no_layers in range(1,20):

            err = 0

            for i in range(iters):

                model = nn.SqrtLorentzianNeuralNetwork(x, y, no_layers*[layer_size], alpha, gamma)

                model.train(10000)

                err += np.sum(np.square(model.predict(x) - y))

            error[layer_size - 1][no_layers - 1] = err/iters

            file.write(str(layer_size) + ',' + str(no_layers) + ';' + str(err/iters) + '\n')

    file.close()

    x = [x for x in range(2,21)]

    a1, = plt.plot(x, error[:][0])
    m1 = "1 hidden layer"
    a2, = plt.plot(x, error[:][1])
    m2 = "2 hidden layers"
    a3, = plt.plot(x, error[:][2])
    m3 = "3 hidden layers"
    a4, = plt.plot(x, error[:][3])
    m4 = "4 hidden layers"
    plt.ylabel("Final Error")
    plt.xlabel("Size of hidden layer(s)")
    plt.title("Final error for different network complexities")
    plt.legend((a1,a2,a3,a4), (m1,m2,m3,m4))
    plt.show()

def evalSpeed():
    x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    model = pnn.FullLorentzianNeuralNetwork(x, y, [5,3], 0.001, 1)
    
    training_iters = 10000
    
    t0 = time.time()
    model.train(training_iters)
    t1 = time.time()

    print(str(t1-t0))

def compareLorentzErrorCurve():
    x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    layers = [8]
    alpha = 0.001
    gamma = 1

    training_iters = 1000
    training_batch = 100

    modelLorentz = bnn.FullLorentzianNeuralNetwork(x, y, layers, alpha, gamma)
    modelDerivLorentz = bnn.FullDerivLorentzianNeuralNetwork(x, y, layers, alpha, gamma)
    
    modelSigmoid = bnn.SigmoidNeuralNetwork(x, y)
    
    m = np.zeros(training_iters)

    repeats = 10

    errorLorentz = np.zeros((training_iters, repeats))
    errorDerivLorentz = np.zeros((training_iters, repeats))
    errorSigmoid = np.zeros((training_iters, repeats))
    # errorTanh = np.zeros(training_iters)
    # errorReLU = np.zeros(training_iters)

    file = open('CompareErrorCurves.txt', 'w')
    file.write("Training Iterations; Lorentz , DerivLorentz , Sigmoid , Tanh , ReLU\n")

    for j in range(repeats):
        for i in range(training_iters):

            errorLorentz[i, j] = np.sum(np.square(modelLorentz.predict(x) - y))
            errorDerivLorentz[i, j] = np.sum(np.square(modelDerivLorentz.predict(x) - y))
            errorSigmoid[i, j] = np.sum(np.square(modelSigmoid.predict(x) - y))

            modelLorentz.train(training_batch)
            modelDerivLorentz.train(training_batch)
            modelSigmoid.train(training_batch)
            # modelTanh.fit(x, y_)
            # modelReLU.fit(x, y_)

            m[i] = i * training_batch


            # file.write(str(m[i]) + ';' + str(errorLorentz[i]) + ',' + str(errorDerivLorentz[i]) + ',' + str(errorSigmoid[i]) + ',' + str(errorTanh[i]) + ',' + str(errorReLU[i]) + '\n')
            
    errorLorentz = np.mean(errorLorentz, axis=1)
    errorDerivLorentz = np.mean(errorDerivLorentz, axis=1)
    errorSigmoid = np.mean(errorSigmoid, axis=1)

    for i in range(training_iters):
        file.write(str(m[i]) + ';' + str(errorLorentz[i]) + ',' + str(errorDerivLorentz[i]) + ',' + str(errorSigmoid[i]) + '\n')

    file.close()

    return


evalLorentzErrorCurve(0.001, 1, [5, 3])



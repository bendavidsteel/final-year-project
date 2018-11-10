# written by ben steel
# 24/10/18
# cma library taken from https://github.com/CMA-ES/pycma

from neuralnetwork import FullLorentzianNeuralNetwork
import numpy as np
# import cma
import matplotlib.pyplot as plt

def evalLorentzError(alpha, gamma):
    x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    layers = [4]

    model = FullLorentzianNeuralNetwork(x, y, layers, alpha, gamma)
    
    m = np.zeros(1000)
    n = np.zeros(1000)

    print(model.predict(x))

    for i in range(1000):
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

    m = np.linspace(-3, 3, 100)
    n = np.zeros(100)

    for i in range(len(m)):

        model = FullLorentzianNeuralNetwork(x, y, layers, alpha, m[i])

        model.train(10000)

        n[i] = np.sum(np.square(model.predict(x) - y))

    plt.plot(m, n)
    plt.axis([0, np.max(m), 0, np.max(n)])
    plt.ylabel("Final Error")
    plt.xlabel("Gamma")
    plt.title("Final error for different values of Gamma")
    plt.show()

print(evalLorentzError(0.001, 0.3))
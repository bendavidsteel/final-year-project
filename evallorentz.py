# written by ben steel
# 24/10/18
# cma library taken from https://github.com/CMA-ES/pycma

from neuralnetwork import NeuralNetwork
import numpy as np
import cma
import matplotlib.pyplot as plt

def evalLorentz(params):
    x = np.array([[0,0],
                [1,0],
                [0,1],
                [1,1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    params[0] = int((2 * params[0])**2 + 1)
    if params[0] > 3:
        params[0] = 3

    model = NeuralNetwork(x, y, params)
    model.train(10000)
    
    output = model.predict(x)

    m = np.linspace(-3, 3, 1000)
    n = np.zeros(1000)

    for i in range(len(m)):
        n[i] = model.activation(m[i])

    plt.plot(m, n)
    plt.show()

    return np.sum(np.square(output - y))

es = cma.CMAEvolutionStrategy(9*[0], 0.5)
es.optimize(evalLorentz)

# print(es.result_pretty())

print(evalLorentz([3, 0.31, -0.27, -0.12, -0.04, -2.67, 0.59]))
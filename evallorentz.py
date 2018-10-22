from neuralnetwork import NeuralNetwork
import numpy as np
import cma

def evalLorentz(params):
    x = np.array([[0,0],
                [1,0],
                [0,1],
                [1,1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    x0 = params[0]
    lambd = params[1]

    model = NeuralNetwork(x, y, x0, lambd)
    model.train(10000)
    
    output = model.predict(x)

    return np.sum(np.square(output - y))

# es = cma.CMAEvolutionStrategy([1], 0.5)
# es.optimize(evalLorentz)

# print(es.result_pretty())

print(evalLorentz([0.043, 0.65]))
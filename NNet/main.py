from neuralnetwork import NeuralNetwork
import numpy as np

x = np.array([[0,0],
              [1,0],
              [0,1],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

x0 = 0
lambd = 1

model = NeuralNetwork(x, y, x0, lambd)
model.train(100000)
print(model.predict(x))

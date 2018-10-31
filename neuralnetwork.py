# 24/10/18
# adapted from james loy's NN implementation, Lorentz functions are mine
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np

class NeuralNetwork:
  def __init__(self, x, y, params):
    self.input = x
    self.weights1 = np.random.rand(self.input.shape[1], 4)
    self.weights2 = np.random.rand(4,1)
    self.y = y
    self.output = np.zeros(self.y.shape)

    #lorentz function parameters
    self.params = params

  def feedForward(self):
    self.layer1 = self.activation(np.dot(self.input, self.weights1))
    self.output = self.activation(np.dot(self.layer1, self.weights2))

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.activation(self.output)))
    d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.derivActivation(self.output), self.weights2.T) * self.derivActivation(self.layer1)))

    # update the weights with the derivative (slope) of the loss function
    self.weights1 += d_weights1
    self.weights2 += d_weights2

  def train(self, iterations):
    #train for number of iterations
    for i in range(iterations):
      self.feedForward()
      self.backProp()

  def predict(self, x):
    #use model
    self.input = x
    self.feedForward()

    return self.output

  def lorentz(self, x, x0, lambd):
    # lorentz function
    return (0.5*lambd)/(np.pi * (np.square(x - x0) + np.square(0.5*lambd)))

  def derivLorentz(self, x, x0, lambd):
    # derivative of lorentz function
    return (-16*lambd*(x - x0))/(np.pi * np.square(4*np.square(x - x0) + lambd**2)) 

  def sigmoid(self, x):
    # sigmoid function
    return 1/(1 + np.exp(-x))

  def derivSigmoid(self, x):
    # derivative of sigmoid function
    return np.exp(-x)/np.square(1 + np.exp(-x))

  def activation(self, x):
    depth = int(self.params[0])

    for i in range(1, (depth*2) + 1, 2):
      x = self.lorentz(x, self.params[i], self.params[i+1])

    return x

  def derivActivation(self, x):
    depth = int(self.params[0])

    for i in range(1, (depth*2) + 1, 2):
      x = self.derivLorentz(x, self.params[i], self.params[i+1])

    return x

    



import numpy as np

class NeuralNetwork:
  def __init__(self, x, y, x0, lambd):
    self.input = x
    self.weights1 = np.random.rand(self.input.shape[1], 4)
    self.weights2 = np.random.rand(4,1)
    self.y = y
    self.output = np.zeros(self.y.shape)

    #lorentz function parameters
    self.x0 = x0
    self.lambd = lambd

  def feedForward(self):
    self.layer1 = self.activation(np.dot(self.input, self.weights1))
    self.output = self.activation(np.dot(self.layer1, self.weights2))

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.activationDeriv(self.output)))
    d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.activationDeriv(self.output), self.weights2.T) * self.activationDeriv(self.layer1)))

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

  def activation(self, x):
    #return 1/(1 + np.exp(-1*x))
    return (0.5*self.lambd)/(np.pi * (np.square(x - self.x0) + np.square(0.5*self.lambd)))

  def activationDeriv(self, x):
    #return np.exp(-1*x)/((1 + np.exp(-1*x))**2)
    return (-16*self.lambd*(x - self.x0))/(np.pi * np.square(4*np.square(x - self.x0) + self.lambd**2)) 


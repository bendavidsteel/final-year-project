# 24/10/18
# adapted from james loy's NN implementation, Lorentz functions are mine
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# 07/11/18
# further adaptation with correctly derived back propagation for using Lorentzian function, with weights being parameters

import numpy as np

class NeuralNetwork:
  def __init__(self, x, y, layers, alpha):

    # hyperparameters
    self.hidden_layer_sizes = layers
    self.alpha = alpha

    # ensuring inputs and targets are np arrays
    self.x = np.asarray(x)
    
    # targets
    self.y = np.asarray(y)

    self.weights_x0 = []
    self.weights_gamma = []
    
    # first hidden layer weights
    self.weights_x0.append(np.random.rand(self.hidden_layer_sizes[0], self.input.shape[1]))
    self.weights_gamma.append(np.random.rand(self.hidden_layer_sizes[0], self.input.shape[1]))

    # additional hidden layer weights
    for i in range(len(self.hidden_layer_sizes)-1):
      self.weights_x0.append(np.random.rand(self.hidden_layer_sizes[i+1], self.hidden_layer_sizes[i]))
      self.weights_gamma.append(np.random.rand(self.hidden_layer_sizes[i+1], self.hidden_layer_sizes[i]))
    
    # output layer weights
    self.weights_x0.append(np.random.rand(self.y.shape[1], self.hidden_layer_sizes[-1]))
    self.weights_gamma.append(np.random.rand(self.y.shape[1], self.hidden_layer_sizes[-1]))

    self.weights_x0 = np.asarray(self.weights_x0)
    self.weights_gamma = np.asarray(self.weights_gamma)

    self.output = np.zeros(self.y.shape)

  def feedForward(self):
    # performing feed forward step
    self.layers = []

    self.layers.append(np.sum(self.lorentz(np.tile(self.input, (self.hidden_layer_sizes[0],1)), self.weights_x0[0], self.weights_gamma[0]), axis=1))

    for i in range(1, len(self.hidden_layer_sizes)):
      self.layers.append(np.sum(self.lorentz(np.tile(self.layers[i-1], (self.hidden_layer_sizes[i],1)), self.weights_x0[i], self.weights_gamma[i]), axis=1))

    self.layers = np.asarray(self.layers)

    self.output = np.sum(self.lorentz(np.tile(self.layers[-1], (self.y.shape[1],1)), self.weights_x0[-1], self.weights_gamma[-1]), axis=1)

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences
    self.delta = [2*(self.y - self.output)]

    self.delta.insert(0, np.matmul(self.lorentzDx(np.tile(self.layers[-1], (self.y.shape[1],1)), self.weights_x0[-1], self.weights_gamma[-1]).T, self.delta[0]))

    # inserting new deltas at front of list
    for i in range(len(self.hidden_layer_sizes)-1, 0, -1):
      self.delta.insert(0, np.matmul(self.lorentzDx(np.tile(self.layers[i-1], (self.hidden_layer_sizes[i],1)), self.weights_x0[i], self.weights_gamma[i]).T, self.delta[0]))

    self.delta = np.asarray(self.delta)
    # finding the derivative with respect to weights
    grad_weights_x0 = [(self.lorentzDx0(np.tile(self.x, (self.hidden_layer_sizes[0],1)), self.weights_x0[0], self.weights_gamma[0]).T * self.delta[0]).T]
    grad_weights_gamma = [(self.lorentzDGamma(np.tile(self.x, (self.hidden_layer_sizes[0],1)), self.weights_x0[0], self.weights_gamma[0]).T * self.delta[0]).T]

    for i in range(len(1, self.delta)):
      grad_weights_x0.append((self.lorentzDx0(np.tile(self.layers[i], (self.hidden_layer_sizes[i],1)), self.weights_x0[i+1], self.weights_gamma[i+1]).T * self.delta[i]).T)
      grad_weights_gamma.append((self.lorentzDGamma(np.tile(self.layers[i], (self.hidden_layer_sizes[i],1)), self.weights_x0[i+1], self.weights_gamma[i+1]).T * self.delta[i]).T)

    grad_weights_x0 = np.asarray(grad_weights_x0)
    grad_weights_gamma = np.asarray(grad_weights_gamma)

    # update the weights with the derivative of the loss function
    self.weights_x0 -= self.alpha * grad_weights_x0
    self.weights_gamma -= self.alpha * grad_weights_gamma


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

  def lorentz(self, x, x0, gamma):
    # lorentz function
    return (0.5*gamma)/(np.pi * np.add(np.square(np.subtract(x , x0)) , np.square(0.5*gamma)))

  def lorentzDx(self, x, x0, gamma):
    # derivative of lorentz function with respect to x
    return -4*np.subtract(x , x0)*(np.pi/gamma)*np.square(self.lorentz(x, x0, gamma))

  def lorentzDx0(self, x, x0, gamma):
    # derivative of lorentz function with respect to x0
    return 4*np.subtract(x , x0)*(np.pi/gamma)*np.square(self.lorentz(x, x0, gamma))

  def lorentzDGamma(self, x, x0, gamma):
    # derivative of lorentz function with respect to gamma
    return (1/gamma)*self.lorentz(x, x0, gamma) - np.square(self.lorentz(x, x0, gamma))

  def sigmoid(self, x):
    # sigmoid function
    return 1/(1 + np.exp(-x))

  def derivSigmoid(self, x):
    # derivative of sigmoid function
    return np.exp(-x)/np.square(1 + np.exp(-x))

  """ def activation(self, x):
    depth = int(self.params[0])

    for i in range(1, (depth*2) + 1, 2):
      x = self.lorentz(x, self.params[i], self.params[i+1])

    return x

  def derivActivation(self, x):
    depth = int(self.params[0])

    for i in range(1, (depth*2) + 1, 2):
      x = self.derivLorentz(x, self.params[i], self.params[i+1])

    return x """

    



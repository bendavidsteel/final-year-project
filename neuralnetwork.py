# 24/10/18
# adapted from james loy's NN implementation, Lorentz functions are mine
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# 07/11/18
# further adaptation with correctly derived back propagation for using Lorentzian function, with weights being parameters

import numpy as np

class FullLorentzianNeuralNetwork:
  def __init__(self, x, y, layers, alpha, gamma):

    # hyperparameters
    self.hidden_layer_sizes = layers
    self.alpha = alpha
    self.gamma = gamma

    # ensuring inputs and targets are np arrays
    self.input = np.asarray(x)
    
    # targets
    self.y = np.asarray(y)

    # parameters
    self.no_samples = y.shape[0]
    self.output_size = y.shape[1]
    self.input_size = self.input.shape[1]

    self.weights_x0 = []
    
    # first hidden layer weights
    self.weights_x0.append(np.random.rand(self.hidden_layer_sizes[0], self.input_size))

    # additional hidden layer weights
    for i in range(len(self.hidden_layer_sizes)-1):
      self.weights_x0.append(np.random.rand(self.hidden_layer_sizes[i+1], self.hidden_layer_sizes[i]))
    
    # output layer weights
    self.weights_x0.append(np.random.rand(self.output_size, self.hidden_layer_sizes[-1]))

    self.weights_x0 = np.asarray(self.weights_x0)

    self.output = np.zeros(self.y.shape)

  def feedForward(self):
    # performing feed forward step
    self.layers = []

    # resizing input and weights for batch processing
    batch_input = np.repeat(self.input, self.hidden_layer_sizes[0], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    batch_weights_x0 = np.tile(self.weights_x0[0], (self.no_samples, 1)).reshape(self.no_samples, self.weights_x0[0].shape[0], self.weights_x0[0].shape[1])
    
    # finding layer output
    self.layers.append(np.sum(self.lorentz(batch_input, batch_weights_x0), axis=2))

    for i in range(1, len(self.hidden_layer_sizes)):
      # resizing
      batch_input = np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], self.layers[i-1].shape[1])
      batch_weights_x0 = np.tile(self.weights_x0[i], (self.layers[i-1].shape[0], 1)).reshape(self.layers[i-1].shape[0], self.weights_x0[i].shape[0], self.weights_x0[i].shape[1])
      
      # finding layer output
      self.layers.append(np.sum(self.lorentz(batch_input, batch_weights_x0), axis=2))

    # resizing
    batch_input = np.repeat(self.layers[-1], self.output_size, axis=0).reshape(self.no_samples, self.output_size, self.layers[-1].shape[1])
    batch_weights_x0 = np.tile(self.weights_x0[-1], (self.layers[-1].shape[0], 1)).reshape(self.layers[-1].shape[0], self.weights_x0[-1].shape[0], self.weights_x0[-1].shape[1])
    
    self.output = np.sum(self.lorentz(batch_input, batch_weights_x0), axis=2)

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences
    delta = [(2*(self.y - self.output)).reshape(self.no_samples, self.output_size, 1)]

    batch_input = np.repeat(self.layers[-1], self.output_size, axis=0).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])
    batch_weights_x0 = np.tile(self.weights_x0[-1], (self.layers[-1].shape[0], 1)).reshape(self.no_samples, self.weights_x0[-1].shape[0], self.weights_x0[-1].shape[1])

    deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
    deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

    delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))
    # inserting new deltas at front of list
    for i in range(len(self.hidden_layer_sizes)-1, 0, -1):
      batch_input = np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      batch_weights_x0 = np.tile(self.weights_x0[i], (self.layers[i-1].shape[0], 1)).reshape(self.no_samples, self.weights_x0[i].shape[0], self.weights_x0[i].shape[1])

      deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
      deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

      delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing
    batch_input = np.repeat(self.input, self.hidden_layer_sizes[0], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    batch_weights_x0 = np.tile(self.weights_x0[0], (self.no_samples, 1)).reshape(self.no_samples, self.weights_x0[0].shape[0], self.weights_x0[0].shape[1])
    
    deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)
    delta_mat = np.tile(delta[0].reshape(self.no_samples, deriv_mat.shape[1], 1), (1,1,deriv_mat.shape[2]))

    grad_weights_x0 = [np.multiply(deriv_mat, delta_mat)]
    
    # for 2nd and more hidden layer weights
    for i in range(1, len(self.hidden_layer_sizes)):
      batch_input = np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      batch_weights_x0 = np.tile(self.weights_x0[i], (self.layers[i-1].shape[0], 1)).reshape(self.layers[i-1].shape[0], self.weights_x0[i].shape[0], self.weights_x0[i].shape[1])
      
      deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)
      delta_mat = np.tile(delta[i].reshape(self.no_samples, deriv_mat.shape[1], 1), (1,1,deriv_mat.shape[2]))

      grad_weights_x0.append(np.multiply(deriv_mat, delta_mat))

    # for output layer weights
    batch_input = np.repeat(self.layers[-1], self.output_size, axis=0).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])
    batch_weights_x0 = np.tile(self.weights_x0[-1], (self.layers[-1].shape[0], 1)).reshape(self.layers[-1].shape[0], self.weights_x0[-1].shape[0], self.weights_x0[-1].shape[1])
    
    deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)
    delta_mat = np.tile(delta[-1].reshape(self.no_samples, deriv_mat.shape[1], 1), (1,1,deriv_mat.shape[2]))

    grad_weights_x0.append(np.multiply(deriv_mat, delta_mat))
    
    grad_weights = [np.divide(np.sum(weights, axis=0),self.no_samples) for weights in grad_weights_x0]

    # update the weights with the derivative of the loss function
    self.weights_x0 += self.alpha * np.asarray(grad_weights)
    

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

  def lorentz(self, x, x0):
    # lorentz function
    return (0.5*self.gamma)/(np.pi * np.add(np.square(np.subtract(x , x0)) , np.square(0.5*self.gamma)))

  def lorentzDx(self, x, x0):
    # derivative of lorentz function with respect to x
    return -4*np.subtract(x , x0)*(np.pi/self.gamma)*np.square(self.lorentz(x, x0))

  def lorentzDx0(self, x, x0):
    # derivative of lorentz function with respect to x0
    return 4*np.subtract(x , x0)*(np.pi/self.gamma)*np.square(self.lorentz(x, x0))

  def sigmoid(self, x):
    # sigmoid function
    return 1/(1 + np.exp(-x))

  def derivSigmoid(self, x):
    # derivative of sigmoid function
    return np.exp(-x)/np.square(1 + np.exp(-x))

    


class SqrtLorentzianNeuralNetwork:
  # models a single spectrometer as the activation function
  # i think this is broken
  def __init__(self, x, y, layers, alpha, gamma):

    # hyperparameters
    self.hidden_layer_sizes = layers
    self.alpha = alpha
    self.gamma = gamma

    # ensuring inputs and targets are np arrays
    self.input = np.asarray(x)
    
    # targets
    self.y = np.asarray(y)

    # parameters
    self.no_samples = y.shape[0]
    self.output_size = y.shape[1]
    self.input_size = self.input.shape[1]

    self.weights_x0 = []
    
    # first hidden layer weights
    self.weights_x0.append(np.random.rand(self.hidden_layer_sizes[0], 1))

    # additional hidden layer weights
    for i in range(len(self.hidden_layer_sizes)-1):
      self.weights_x0.append(np.random.rand(self.hidden_layer_sizes[i+1], 1))
    
    # output layer weights
    self.weights_x0.append(np.random.rand(self.output_size, 1))

    self.weights_x0 = np.asarray(self.weights_x0)

    self.output = np.zeros(self.y.shape)

  def feedForward(self):
    # performing feed forward step
    self.layers = []

    # resizing input and weights for batch processing
    batch_input = np.sum(np.repeat(self.input, self.hidden_layer_sizes[0], axis=0), axis=1).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)
    batch_weights_x0 = np.tile(self.weights_x0[0], (self.input.shape[0], 1)).reshape(self.no_samples, self.weights_x0[0].shape[0], self.weights_x0[0].shape[1])
    
    # finding layer output
    self.layers.append(self.lorentz(batch_input, batch_weights_x0))

    for i in range(1, len(self.hidden_layer_sizes)):
      # resizing
      batch_input = np.sum(np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0), axis=1).reshape(self.no_samples, self.hidden_layer_sizes[i], 1)
      batch_weights_x0 = np.tile(self.weights_x0[i], (self.layers[i-1].shape[0], 1)).reshape(self.layers[i-1].shape[0], self.weights_x0[i].shape[0], self.weights_x0[i].shape[1])
      
      # finding layer output
      self.layers.append(self.lorentz(batch_input, batch_weights_x0))

    # resizing
    batch_input = np.sum(np.repeat(self.layers[-1], self.output_size, axis=0), axis=1).reshape(self.no_samples, self.output_size, 1)
    batch_weights_x0 = np.tile(self.weights_x0[-1], (self.layers[-1].shape[0], 1)).reshape(self.layers[-1].shape[0], self.weights_x0[-1].shape[0], self.weights_x0[-1].shape[1])
    
    self.output = self.lorentz(batch_input, batch_weights_x0)

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences
    target = self.y.reshape(self.no_samples, self.output_size, 1)
    delta = [(2*(target - self.output)).reshape(self.no_samples, self.output_size, 1)]

    delta.insert(0, np.asarray([np.matmul(np.ones((self.hidden_layer_sizes[-1], self.output_size)), delta) for delta in delta[0]]))
    # inserting new deltas at front of list
    for i in range(len(self.hidden_layer_sizes)-1, 0, -1):
      delta.insert(0, np.asarray([np.matmul(np.ones((self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i])), delta) for delta in delta[0]]))

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing
    batch_input = np.repeat(np.sum(self.input, axis=1), self.hidden_layer_sizes[0], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)
    batch_weights_x0 = np.tile(self.weights_x0[0], (self.no_samples, 1)).reshape(self.no_samples, self.weights_x0[0].shape[0], self.weights_x0[0].shape[1])
    
    deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)
    delta_mat = np.tile(delta[0].reshape(self.no_samples, deriv_mat.shape[1], 1), (1,1,deriv_mat.shape[2]))

    grad_weights_x0 = [np.multiply(deriv_mat, delta_mat)]
    
    # for 2nd and more hidden layer weights
    for i in range(1, len(self.hidden_layer_sizes)):
      batch_input = np.repeat(np.sum(self.layers[i-1], axis=1), self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], 1)
      batch_weights_x0 = np.tile(self.weights_x0[i], (self.layers[i-1].shape[0], 1)).reshape(self.layers[i-1].shape[0], self.weights_x0[i].shape[0], self.weights_x0[i].shape[1])
      
      deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)
      delta_mat = np.tile(delta[i].reshape(self.no_samples, deriv_mat.shape[1], 1), (1,1,deriv_mat.shape[2]))

      grad_weights_x0.append(np.multiply(deriv_mat, delta_mat))

    # for output layer weights
    batch_input = np.repeat(np.sum(self.layers[-1], axis=1), self.output_size, axis=0).reshape(self.no_samples, self.output_size, 1)
    batch_weights_x0 = np.tile(self.weights_x0[-1], (self.layers[-1].shape[0], 1)).reshape(self.layers[-1].shape[0], self.weights_x0[-1].shape[0], self.weights_x0[-1].shape[1])
    
    deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)
    delta_mat = np.tile(delta[-1].reshape(self.no_samples, deriv_mat.shape[1], 1), (1,1,deriv_mat.shape[2]))

    grad_weights_x0.append(np.multiply(deriv_mat, delta_mat))
    
    grad_weights = [np.divide(np.sum(weights, axis=0),self.no_samples) for weights in grad_weights_x0]

    # update the weights with the derivative of the loss function
    self.weights_x0 += self.alpha * np.asarray(grad_weights)
    

  def train(self, iterations):
    #train for number of iterations
    for i in range(iterations):
      self.feedForward()
      self.backProp()

  def predict(self, x):
    #use model
    self.input = x
    self.feedForward()

    return self.output.reshape(self.no_samples, self.output_size)

  def lorentz(self, x, x0):
    # lorentz function
    return (0.5*self.gamma)/(np.pi * np.add(np.square(np.subtract(x , x0)) , np.square(0.5*self.gamma)))

  def lorentzDx(self, x, x0):
    # derivative of lorentz function with respect to x
    return -4*np.subtract(x , x0)*(np.pi/self.gamma)*np.square(self.lorentz(x, x0))

  def lorentzDx0(self, x, x0):
    # derivative of lorentz function with respect to x0
    return 4*np.subtract(x , x0)*(np.pi/self.gamma)*np.square(self.lorentz(x, x0))


class LorentzianInNeuralNetwork:
  # models 1 spectrometer going into each node
  def __init__(self, x, y, layers, alpha, gamma):

    # hyperparameters
    self.hidden_layer_sizes = layers
    self.alpha = alpha
    self.gamma = gamma

    # ensuring inputs and targets are np arrays
    self.input = np.asarray(x)
    
    # targets
    self.y = np.asarray(y)

    # parameters
    self.no_samples = y.shape[0]
    self.output_size = y.shape[1]
    self.input_size = self.input.shape[1]

    self.weights_x0 = []
    
    # first hidden layer weights
    self.weights_x0.append(np.random.rand(self.hidden_layer_sizes[0], 1))

    # additional hidden layer weights
    for i in range(len(self.hidden_layer_sizes)-1):
      self.weights_x0.append(np.random.rand(self.hidden_layer_sizes[i+1], 1))
    
    # output layer weights
    self.weights_x0.append(np.random.rand(self.output_size, 1))

    self.weights_x0 = np.asarray(self.weights_x0)

    self.output = np.zeros(self.y.shape)

  def feedForward(self):
    # performing feed forward step
    self.layers = []

    # resizing input and weights for batch processing
    batch_input = np.repeat(self.input, self.hidden_layer_sizes[0], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    batch_weights_x0 = np.tile(np.repeat(self.weights_x0[0], self.input_size, axis=1), (self.no_samples, 1)).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    
    # finding layer output
    self.layers.append(np.sum(self.lorentz(batch_input, batch_weights_x0), axis=2))

    for i in range(1, len(self.hidden_layer_sizes)):
      # resizing
      batch_input = np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      batch_weights_x0 = np.tile(np.repeat(self.weights_x0[i], self.hidden_layer_sizes[i-1], axis=1), (self.layers[i-1].shape[0], 1)).reshape(self.layers[i-1].shape[0], self.hidden_layer_sizes[i], self.layers[i-1].shape[1])
      
      # finding layer output
      self.layers.append(np.sum(self.lorentz(batch_input, batch_weights_x0), axis=2))

    # resizing
    batch_input = np.repeat(self.layers[-1], self.output_size, axis=0).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])
    batch_weights_x0 = np.tile(np.repeat(self.weights_x0[-1], self.hidden_layer_sizes[-1], axis=1), (self.layers[-1].shape[0], 1)).reshape(self.layers[-1].shape[0], self.output_size, self.layers[-1].shape[1])
    
    self.output = np.sum(self.lorentz(batch_input, batch_weights_x0), axis=2)

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences
    delta = [(2*(self.y - self.output)).reshape(self.no_samples, self.output_size, 1)]

    batch_input = np.repeat(self.layers[-1], self.output_size, axis=0).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])
    batch_weights_x0 = np.tile(np.repeat(self.weights_x0[-1], self.hidden_layer_sizes[-1], axis=1), (self.layers[-1].shape[0], 1)).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])

    deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
    deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

    delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))
    # inserting new deltas at front of list
    for i in range(len(self.hidden_layer_sizes)-1, 0, -1):
      batch_input = np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      batch_weights_x0 = np.tile(np.repeat(self.weights_x0[i], self.hidden_layer_sizes[i-1], axis=1), (self.layers[i-1].shape[0], 1)).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])

      deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
      deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

      delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing
    batch_input = np.repeat(self.input, self.hidden_layer_sizes[0], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    batch_weights_x0 = np.tile(np.repeat(self.weights_x0[0], self.input_size, axis=1), (self.no_samples, 1)).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    
    deriv_mat = np.sum(self.lorentzDx0(batch_input, batch_weights_x0), axis=2).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)
    delta_mat = np.tile(delta[0].reshape(self.no_samples, deriv_mat.shape[1], 1), (1,1,deriv_mat.shape[2]))

    grad_weights_x0 = [np.multiply(deriv_mat, delta_mat)]
    
    # for 2nd and more hidden layer weights
    for i in range(1, len(self.hidden_layer_sizes)):
      batch_input = np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      batch_weights_x0 = np.tile(np.repeat(self.weights_x0[i], self.hidden_layer_sizes[i-1], axis=1), (self.layers[i-1].shape[0], 1)).reshape(self.layers[i-1].shape[0], self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      
      deriv_mat = np.sum(self.lorentzDx0(batch_input, batch_weights_x0), axis=2).reshape(self.no_samples, self.hidden_layer_sizes[i], 1)
      delta_mat = np.tile(delta[i].reshape(self.no_samples, deriv_mat.shape[1], 1), (1,1,deriv_mat.shape[2]))

      grad_weights_x0.append(np.multiply(deriv_mat, delta_mat))

    # for output layer weights
    batch_input = np.repeat(self.layers[-1], self.output_size, axis=0).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])
    batch_weights_x0 = np.tile(np.repeat(self.weights_x0[-1], self.hidden_layer_sizes[-1], axis=1), (self.layers[-1].shape[0], 1)).reshape(self.layers[-1].shape[0], self.output_size, self.hidden_layer_sizes[-1])
    
    deriv_mat = np.sum(self.lorentzDx0(batch_input, batch_weights_x0), axis=2).reshape(self.no_samples, self.output_size, 1)
    delta_mat = np.tile(delta[-1].reshape(self.no_samples, deriv_mat.shape[1], 1), (1,1,deriv_mat.shape[2]))

    grad_weights_x0.append(np.multiply(deriv_mat, delta_mat))
    
    grad_weights = [np.divide(np.sum(weights, axis=0),self.no_samples) for weights in grad_weights_x0]

    # update the weights with the derivative of the loss function
    self.weights_x0 += self.alpha * np.asarray(grad_weights)
    

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

  def lorentz(self, x, x0):
    # lorentz function
    return (0.5*self.gamma)/(np.pi * np.add(np.square(np.subtract(x , x0)) , np.square(0.5*self.gamma)))

  def lorentzDx(self, x, x0):
    # derivative of lorentz function with respect to x
    return -4*np.subtract(x , x0)*(np.pi/self.gamma)*np.square(self.lorentz(x, x0))

  def lorentzDx0(self, x, x0):
    # derivative of lorentz function with respect to x0
    return 4*np.subtract(x , x0)*(np.pi/self.gamma)*np.square(self.lorentz(x, x0))

  def sigmoid(self, x):
    # sigmoid function
    return 1/(1 + np.exp(-x))

  def derivSigmoid(self, x):
    # derivative of sigmoid function
    return np.exp(-x)/np.square(1 + np.exp(-x))



class LorentzianInOutNeuralNetwork:
  # models 2 spectrometers going into and out of each node
  def __init__(self, x, y, layers, alpha, gamma):

    # hyperparameters
    self.hidden_layer_sizes = layers
    self.alpha = alpha
    self.gamma = gamma

    # ensuring inputs and targets are np arrays
    self.input = np.asarray(x)
    
    # targets
    self.y = np.asarray(y)

    # parameters
    self.no_samples = y.shape[0]
    self.output_size = y.shape[1]
    self.input_size = self.input.shape[1]

    self.weights_x0k = []
    self.weights_x0j = []
    
    # first hidden layer weights
    self.weights_x0k.append(np.random.rand(self.hidden_layer_sizes[0], 1))
    self.weights_x0j.append(np.random.rand(1, self.input_size))

    # additional hidden layer weights
    for i in range(len(self.hidden_layer_sizes)-1):
      self.weights_x0k.append(np.random.rand(self.hidden_layer_sizes[i+1], 1))
      self.weights_x0j.append(np.random.rand(1, self.hidden_layer_sizes[i]))
    
    # output layer weights
    self.weights_x0k.append(np.random.rand(self.output_size, 1))
    self.weights_x0j.append(np.random.rand(1, self.hidden_layer_sizes[-1]))

    self.output = np.zeros(self.y.shape)

  def feedForward(self):
    # performing feed forward step
    self.layers = []

    # resizing input and weights for batch processing
    batch_input = np.repeat(self.input, self.hidden_layer_sizes[0], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    batch_weights_x0j = np.tile(np.tile(self.weights_x0j[0], (self.no_samples, 1)), (self.hidden_layer_sizes[0],1)).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)

    batch_inter = self.lorentz(batch_input, batch_weights_x0j)
    batch_weights_x0k = np.tile(np.repeat(self.weights_x0k[0], self.input_size, axis=1), (self.no_samples, 1)).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    
    # finding layer output
    self.layers.append(np.sum(self.lorentz(batch_inter, batch_weights_x0k), axis=2))

    for i in range(1, len(self.hidden_layer_sizes)):
      # resizing
      batch_input = np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      batch_weights_x0j = np.tile(np.tile(self.weights_x0j[i], (self.no_samples, 1)), (self.hidden_layer_sizes[i], 1)).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])

      batch_inter = self.lorentz(batch_input, batch_weights_x0j)
      batch_weights_x0k = np.tile(np.repeat(self.weights_x0k[i], self.hidden_layer_sizes[i-1], axis=1), (self.layers[i-1].shape[0], 1)).reshape(self.layers[i-1].shape[0], self.hidden_layer_sizes[i], self.layers[i-1].shape[1])
      
      # finding layer output
      self.layers.append(np.sum(self.lorentz(batch_inter, batch_weights_x0k), axis=2))

    # resizing
    batch_input = np.repeat(self.layers[-1], self.output_size, axis=0).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])
    batch_weights_x0j = np.tile(np.tile(self.weights_x0j[-1], (self.no_samples, 1)), (self.output_size, 1)).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])

    batch_inter = self.lorentz(batch_input, batch_weights_x0j)
    batch_weights_x0k = np.tile(np.repeat(self.weights_x0k[-1], self.hidden_layer_sizes[-1], axis=1), (self.layers[-1].shape[0], 1)).reshape(self.layers[-1].shape[0], self.output_size, self.layers[-1].shape[1])
    
    self.output = np.sum(self.lorentz(batch_input, batch_weights_x0k), axis=2)

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences
    delta = [(2*(self.y - self.output)).reshape(self.no_samples, self.output_size, 1)]

    #chain rule
    batch_input = np.repeat(self.layers[-1], self.output_size, axis=0).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])
    batch_weights_x0j = np.tile(np.tile(self.weights_x0j[-1], (self.no_samples, 1)), (self.output_size, 1)).reshape(self.layers[-1].shape[0], self.output_size, self.layers[-1].shape[1])

    j_mat = self.lorentz(batch_input, batch_weights_x0j)
    deriv_j_mat = self.lorentzDx(batch_input, batch_weights_x0j)

    batch_weights_x0k = np.tile(np.repeat(self.weights_x0k[-1], self.hidden_layer_sizes[-1], axis=1), (self.layers[-1].shape[0], 1)).reshape(self.layers[-1].shape[0], self.output_size, self.layers[-1].shape[1])

    deriv_k_mat = self.lorentzDx(j_mat, batch_weights_x0k)

    deriv_mat = np.multiply(deriv_k_mat, deriv_j_mat)

    deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

    delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))
    # inserting new deltas at front of list
    for i in range(len(self.hidden_layer_sizes)-1, 0, -1):
      batch_input = np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      batch_weights_x0j = np.tile(np.tile(self.weights_x0j[i], (self.no_samples, 1)), (self.hidden_layer_sizes[i], 1)).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])

      j_mat = self.lorentz(batch_input, batch_weights_x0j)
      deriv_j_mat = self.lorentzDx(batch_input, batch_weights_x0j)

      batch_weights_x0k = np.tile(np.repeat(self.weights_x0k[i], self.hidden_layer_sizes[i-1], axis=1), (self.layers[i-1].shape[0], 1)).reshape(self.layers[i-1].shape[0], self.hidden_layer_sizes[i], self.layers[i-1].shape[1])
      
      deriv_k_mat = self.lorentzDx(j_mat, batch_weights_x0k)

      deriv_mat = np.multiply(deriv_k_mat, deriv_j_mat)
      
      deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

      delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing
    batch_input = np.repeat(self.input, self.hidden_layer_sizes[0], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    batch_weights_x0j = np.tile(np.tile(self.weights_x0j[0], (self.no_samples, 1)), (self.hidden_layer_sizes[0], 1)).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)

    inter = self.lorentz(batch_input, batch_weights_x0j)
    inter_deriv_x0 = self.lorentzDx0(batch_input, batch_weights_x0j)

    batch_weights_x0k = np.tile(np.repeat(self.weights_x0k[0], self.input_size, axis=1), (self.no_samples, 1)).reshape(self.no_samples, self.hidden_layer_sizes[0], self.input_size)
    
    deriv = np.sum(self.lorentzDx(inter, batch_weights_x0k), axis=2).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)
    deriv_x0 = np.sum(self.lorentzDx0(inter, batch_weights_x0k), axis=2).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)

    # deriv_mat = np.sum(self.lorentzDx0(batch_input, batch_weights_x0), axis=2).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)
    delta_mat = np.tile(delta[0], (1,1,self.input_size))

    batch_grad_weights_x0k = [np.multiply(deriv_x0, delta[0])]

    grad_weights_inter = np.multiply(np.tile(np.multiply(deriv, delta[0]), (1,1, self.input_size)), inter_deriv_x0)

    batch_grad_weights_x0j = [np.sum(grad_weights_inter, axis=1)/self.hidden_layer_sizes[0]]
    
    # for 2nd and more hidden layer weights
    for i in range(1, len(self.hidden_layer_sizes)):
      batch_input = np.repeat(self.layers[i-1], self.hidden_layer_sizes[i], axis=0).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      batch_weights_x0j = np.tile(np.tile(self.weights_x0j[i], (self.no_samples, 1)), (self.hidden_layer_sizes[i], 1)).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])

      inter = self.lorentz(batch_input, batch_weights_x0j)
      inter_deriv_x0 = self.lorentzDx0(batch_input, batch_weights_x0j)

      batch_weights_x0k = np.tile(np.repeat(self.weights_x0k[i], self.hidden_layer_sizes[i-1], axis=1), (self.no_samples, 1)).reshape(self.no_samples, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      
      deriv = np.sum(self.lorentzDx(inter, batch_weights_x0k), axis=2).reshape(self.no_samples, self.hidden_layer_sizes[i], 1)
      deriv_x0 = np.sum(self.lorentzDx0(inter, batch_weights_x0k), axis=2).reshape(self.no_samples, self.hidden_layer_sizes[i], 1)

      # deriv_mat = np.sum(self.lorentzDx0(batch_input, batch_weights_x0), axis=2).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)
      delta_mat = np.tile(delta[i], (1,1,self.hidden_layer_sizes[i-1]))

      batch_grad_weights_x0k.append(np.multiply(deriv_x0, delta[i]))

      grad_weights_inter = np.multiply(np.tile(np.multiply(deriv, delta[i]), (1,1, self.hidden_layer_sizes[i-1])), inter_deriv_x0)

      batch_grad_weights_x0j.append(np.sum(grad_weights_inter, axis=1)/self.hidden_layer_sizes[i])

    # for output layer weights
    batch_input = np.repeat(self.layers[-1], self.output_size, axis=0).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])
    batch_weights_x0j = np.tile(np.tile(self.weights_x0j[-1], (self.no_samples, 1)), (self.output_size, 1)).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])

    inter = self.lorentz(batch_input, batch_weights_x0j)
    inter_deriv_x0 = self.lorentzDx0(batch_input, batch_weights_x0j)

    batch_weights_x0k = np.tile(np.repeat(self.weights_x0k[-1], self.hidden_layer_sizes[-1], axis=1), (self.no_samples, 1)).reshape(self.no_samples, self.output_size, self.hidden_layer_sizes[-1])
    
    deriv = np.sum(self.lorentzDx(inter, batch_weights_x0k), axis=2).reshape(self.no_samples, self.output_size, 1)
    deriv_x0 = np.sum(self.lorentzDx0(inter, batch_weights_x0k), axis=2).reshape(self.no_samples, self.output_size, 1)

    # deriv_mat = np.sum(self.lorentzDx0(batch_input, batch_weights_x0), axis=2).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)
    delta_mat = np.tile(delta[-1], (1,1,self.hidden_layer_sizes[-1]))

    batch_grad_weights_x0k.append(np.multiply(deriv_x0, delta[-1]))

    grad_weights_inter = np.multiply(np.tile(np.multiply(deriv, delta[-1]), (1,1, self.hidden_layer_sizes[-1])), inter_deriv_x0)

    batch_grad_weights_x0j.append(np.sum(grad_weights_inter, axis=1)/self.output_size)
    
    grad_weights_x0k = [np.divide(np.sum(weights, axis=0),self.no_samples) for weights in batch_grad_weights_x0k]
    grad_weights_x0j = [np.divide(np.sum(weights, axis=0),self.no_samples) for weights in batch_grad_weights_x0j]

    # update the weights with the derivative of the loss function
    self.weights_x0k += self.alpha * np.asarray(grad_weights_x0k)
    self.weights_x0j += self.alpha * np.asarray(grad_weights_x0j)
    

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

  def lorentz(self, x, x0):
    # lorentz function
    return (0.5*self.gamma)/(np.pi * np.add(np.square(np.subtract(x , x0)) , np.square(0.5*self.gamma)))

  def lorentzDx(self, x, x0):
    # derivative of lorentz function with respect to x
    return -4*np.subtract(x , x0)*(np.pi/self.gamma)*np.square(self.lorentz(x, x0))

  def lorentzDx0(self, x, x0):
    # derivative of lorentz function with respect to x0
    return 4*np.subtract(x , x0)*(np.pi/self.gamma)*np.square(self.lorentz(x, x0))

  def sigmoid(self, x):
    # sigmoid function
    return 1/(1 + np.exp(-x))

  def derivSigmoid(self, x):
    # derivative of sigmoid function
    return np.exp(-x)/np.square(1 + np.exp(-x))
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
    batch_input = self.input.reshape(self.no_samples, 1, self.input_size)
    batch_weights_x0 = self.weights_x0[0].reshape(1, self.hidden_layer_sizes[0], self.input_size)
    
    # finding layer output
    self.layers.append(np.sum(self.lorentz(batch_input, batch_weights_x0), axis=2))

    for i in range(1, len(self.hidden_layer_sizes)):
      # resizing
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].reshape(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      
      # finding layer output
      self.layers.append(np.sum(self.lorentz(batch_input, batch_weights_x0), axis=2))

    # resizing
    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])
    
    self.output = np.sum(self.lorentz(batch_input, batch_weights_x0), axis=2)

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences
    delta = [(2*(self.y - self.output)).reshape(self.no_samples, self.output_size, 1)]

    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])

    deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
    deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

    delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))
    # inserting new deltas at front of list
    for i in range(len(self.hidden_layer_sizes)-1, 0, -1):
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].reshape(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])

      deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
      deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

      delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing
    batch_input = self.input.reshape(self.no_samples, 1, self.input_size)
    batch_weights_x0 = self.weights_x0[0].reshape(1, self.hidden_layer_sizes[0], self.input_size)
    
    deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)

    grad_weights_x0 = [np.multiply(deriv_mat, delta[0])]
    
    # for 2nd and more hidden layer weights
    for i in range(1, len(self.hidden_layer_sizes)):
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].reshape(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      
      deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)

      grad_weights_x0.append(np.multiply(deriv_mat, delta[i]))

    # for output layer weights
    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])
    
    deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)

    grad_weights_x0.append(np.multiply(deriv_mat, delta[-1]))
    
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





class FullDerivLorentzianNeuralNetwork:
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
    batch_input = self.input.reshape(self.no_samples, 1, self.input_size)
    batch_weights_x0 = self.weights_x0[0].reshape(1, self.hidden_layer_sizes[0], self.input_size)
    
    # finding layer output
    self.layers.append(np.sum(self.lorentzDx(batch_input, batch_weights_x0), axis=2))

    for i in range(1, len(self.hidden_layer_sizes)):
      # resizing
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].reshape(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      
      # finding layer output
      self.layers.append(np.sum(self.lorentzDx(batch_input, batch_weights_x0), axis=2))

    # resizing
    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])
    
    self.output = np.sum(self.lorentzDx(batch_input, batch_weights_x0), axis=2)

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences
    delta = [(2*(self.y - self.output)).reshape(self.no_samples, self.output_size, 1)]

    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])

    deriv_mat = self.lorentzD2x(batch_input, batch_weights_x0)
    deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

    delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))
    # inserting new deltas at front of list
    for i in range(len(self.hidden_layer_sizes)-1, 0, -1):
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].reshape(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])

      deriv_mat = self.lorentzD2x(batch_input, batch_weights_x0)
      deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

      delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing
    batch_input = self.input.reshape(self.no_samples, 1, self.input_size)
    batch_weights_x0 = self.weights_x0[0].reshape(1, self.hidden_layer_sizes[0], self.input_size)
    
    deriv_mat = self.lorentzD2x0(batch_input, batch_weights_x0)

    grad_weights_x0 = [np.multiply(deriv_mat, delta[0])]
    
    # for 2nd and more hidden layer weights
    for i in range(1, len(self.hidden_layer_sizes)):
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].reshape(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      
      deriv_mat = self.lorentzD2x0(batch_input, batch_weights_x0)

      grad_weights_x0.append(np.multiply(deriv_mat, delta[i]))

    # for output layer weights
    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])
    
    deriv_mat = self.lorentzD2x0(batch_input, batch_weights_x0)

    grad_weights_x0.append(np.multiply(deriv_mat, delta[-1]))
    
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

  def lorentzD2x(self, x, x0):
    return 2*(np.pi**2)*(self.lorentz(x, x0)**3)*(((12/(self.gamma**2))*((x - x0)**2)) - 1)

  def lorentzD2x0(self, x, x0):
      return -1*self.lorentzD2x(x, x0)

  def sigmoid(self, x):
    # sigmoid function
    return 1/(1 + np.exp(-x))

  def derivSigmoid(self, x):
    # derivative of sigmoid function
    return np.exp(-x)/np.square(1 + np.exp(-x))
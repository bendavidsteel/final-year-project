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


class SigmoidNeuralNetwork:
  def __init__(self, x, y):
      self.input      = x
      self.weights1   = np.random.rand(self.input.shape[1],4) 
      self.weights2   = np.random.rand(4,1)                 
      self.y          = y
      self.output     = np.zeros(self.y.shape)

  def feedforward(self):
      self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
      self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

  def backprop(self):
      # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
      d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
      d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

      # update the weights with the derivative (slope) of the loss function
      self.weights1 += d_weights1
      self.weights2 += d_weights2

  def train(self, iterations):
    #train for number of iterations
    for i in range(iterations):
      self.feedforward()
      self.backprop()

  def predict(self, x):
    #use model
    self.input = x
    self.feedforward()

    return self.output

  def sigmoid(self, x):
    # sigmoid function
    return 1/(1 + np.exp(-x))

  def sigmoid_derivative(self, x):
    # derivative of sigmoid function
    return np.exp(-x)/np.square(1 + np.exp(-x))



class TanhNeuralNetwork:
  def __init__(self, x, y):
      self.input      = x
      self.weights1   = np.random.rand(self.input.shape[1],4) 
      self.weights2   = np.random.rand(4,1)                 
      self.y          = y
      self.output     = np.zeros(self.y.shape)

  def feedforward(self):
      self.layer1 = self.tanh(np.dot(self.input, self.weights1))
      self.output = self.tanh(np.dot(self.layer1, self.weights2))

  def backprop(self):
      # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
      d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.tanh_deriv(self.output)))
      d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.tanh_deriv(self.output), self.weights2.T) * self.tanh_deriv(self.layer1)))

      # update the weights with the derivative (slope) of the loss function
      self.weights1 += d_weights1
      self.weights2 += d_weights2

  def train(self, iterations):
    #train for number of iterations
    for i in range(iterations):
      self.feedforward()
      self.backprop()

  def predict(self, x):
    #use model
    self.input = x
    self.feedforward()

    return self.output

  def tanh(self, x):
    return np.tanh(x)

  def tanh_deriv(self, x):
    return 1.0 - np.tanh(x)**2


class reluNeuralNetwork:
  def __init__(self, x, y):
      self.input      = x
      self.weights1   = np.random.rand(self.input.shape[1],4) 
      self.weights2   = np.random.rand(4,1)                 
      self.y          = y
      self.output     = np.zeros(self.y.shape)

  def feedforward(self):
      self.layer1 = self.relu(np.dot(self.input, self.weights1))
      self.output = self.relu(np.dot(self.layer1, self.weights2))

  def backprop(self):
      # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
      d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.reluDerivative(self.output)))
      d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.reluDerivative(self.output), self.weights2.T) * self.reluDerivative(self.layer1)))

      # update the weights with the derivative (slope) of the loss function
      self.weights1 += d_weights1
      self.weights2 += d_weights2

  def train(self, iterations):
    #train for number of iterations
    for i in range(iterations):
      self.feedforward()
      self.backprop()

  def predict(self, x):
    #use model
    self.input = x
    self.feedforward()

    return self.output

  def relu(self, x):
    x[x < 0] = 0

    return x

  def reluDerivative(self, x):
    x[x <= 0] = 0
    x[x > 0] = 1

    return x





# Needs extra light sources, only difference between this and normal is lorentzian activation
class ActivationLorentzianNeuralNetwork:
  def __init__(self, x, y, layers, alpha=0.001, gamma=1, x0=0):

    # hyperparameters
    self.hidden_layer_sizes = layers
    self.alpha = alpha
    self.gamma = gamma
    self.x0 = x0

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
    self.layers.append(self.lorentz(np.sum(batch_input * batch_weights_x0, axis=2)))

    for i in range(1, len(self.hidden_layer_sizes)):
      # resizing
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].reshape(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      
      # finding layer output
      self.layers.append(self.lorentz(np.sum(batch_input * batch_weights_x0, axis=2)))

    # resizing
    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])
    
    self.output = self.lorentz(np.sum(batch_input * batch_weights_x0, axis=2))

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences

    #output cost

    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])

    delta = [(2*(self.y - self.output)*self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2))).reshape(self.no_samples, self.output_size, 1)]

    # second to last layer delta

    if len(self.hidden_layer_sizes) > 1:
      batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])
        
      weights_T = np.asarray([mat.T for mat in batch_weights_x0])

      weight_delta = np.asarray([np.matmul(mat, delta) for mat, delta in zip(weights_T, delta[0])])

      batch_input = self.layers[-2].reshape(self.no_samples, 1, self.hidden_layer_sizes[-2])
      batch_weights_x0 = self.weights_x0[-2].reshape(1, self.hidden_layer_sizes[-1], self.hidden_layer_sizes[-2])

      deriv_mat = self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2)).reshape(self.no_samples, self.hidden_layer_sizes[-1], 1)

      delta.insert(0, weight_delta * deriv_mat)

      # deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
      # deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

      # delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))
      # inserting new deltas at front of list
      for i in range(len(self.hidden_layer_sizes)-2, 0, -1):
        batch_weights_x0 = self.weights_x0[i+1].reshape(1, self.hidden_layer_sizes[i+1], self.hidden_layer_sizes[i])
        
        weights_T = np.asarray([mat.T for mat in batch_weights_x0])

        weight_delta = np.asarray([np.matmul(mat, delta) for mat, delta in zip(weights_T, delta[0])])

        batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])
        batch_weights_x0 = self.weights_x0[i].reshape(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])

        deriv_mat = self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2)).reshape(self.no_samples, self.hidden_layer_sizes[i], 1)

        delta.insert(0, weight_delta * deriv_mat)

      # finding delta for first layer
      batch_weights_x0 = self.weights_x0[1].reshape(1, self.hidden_layer_sizes[1], self.hidden_layer_sizes[0])
        
      weights_T = np.asarray([mat.T for mat in batch_weights_x0])

      weight_delta = np.asarray([np.matmul(mat, delta) for mat, delta in zip(weights_T, delta[0])])

      batch_input = self.input.reshape(self.no_samples, 1, self.input_size)
      batch_weights_x0 = self.weights_x0[0].reshape(1, self.hidden_layer_sizes[0], self.input_size)

      deriv_mat = self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2)).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)

      delta.insert(0, weight_delta * deriv_mat)

    else:

      batch_weights_x0 = self.weights_x0[1].reshape(1, self.output_size, self.hidden_layer_sizes[0])
        
      weights_T = np.asarray([mat.T for mat in batch_weights_x0])

      weight_delta = np.asarray([np.matmul(mat, delta) for mat, delta in zip(weights_T, delta[0])])

      batch_input = self.input.reshape(self.no_samples, 1, self.input_size)
      batch_weights_x0 = self.weights_x0[0].reshape(1, self.hidden_layer_sizes[0], self.input_size)

      deriv_mat = self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2)).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)

      delta.insert(0, weight_delta * deriv_mat)

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing
    batch_input = self.input.reshape(self.no_samples, 1, self.input_size)

    grad_weights_x0 = [np.multiply(batch_input, delta[0])]
    
    # for 2nd and more hidden layer weights
    for i in range(1, len(self.hidden_layer_sizes)):
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])

      grad_weights_x0.append(np.multiply(batch_input, delta[i]))

    # for output layer weights
    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])

    grad_weights_x0.append(np.multiply(batch_input, delta[-1]))
    
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

  def lorentz(self, x):
    # lorentz function
    return (0.5*self.gamma)/(np.pi * np.add(np.square(np.subtract(x , self.x0)) , np.square(0.5*self.gamma)))

  def lorentzDx(self, x):
    # derivative of lorentz function with respect to x
    return -4*np.subtract(x , self.x0)*(np.pi/self.gamma)*np.square(self.lorentz(x))



# Needs extra light sources, normal weights with tunable activation function, (effectively bias weight)
class TunableLorentzianNeuralNetwork:
  def __init__(self, x, y, layers, alpha=0.001, gamma=1):

    # hyperparameters
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

    self.layer_sizes = [self.input_size] + layers + [self.output_size]

    self.weights = []

    # additional hidden layer weights
    for i in range(len(self.layer_sizes) - 1):
      self.weights.append(np.random.rand(self.layer_sizes[i+1], self.layer_sizes[i]) - 0.5)

    self.weights = np.asarray(self.weights)

    self.bias = []

    for i in range(1, len(self.layer_sizes)):
      self.bias.append(np.random.rand(self.layer_sizes[i]) - 0.5)

    self.bias = np.asarray(self.bias)

    self.output = np.zeros(self.y.shape)

  def feedForward(self):
    # performing feed forward step
    self.layers = []

    # resizing input and weights for batch processing
    batch_input = self.input.reshape(self.no_samples, 1, self.layer_sizes[0])
    batch_weights = self.weights[0].reshape(1, self.layer_sizes[1], self.layer_sizes[0])
    
    # finding layer output
    self.layers.append(self.lorentz(np.sum(batch_input * batch_weights_x0, axis=2), self.bias[0]))

    for i in range(1, len(self.layer_sizes)-1):
      # resizing
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.layer_sizes[i])
      batch_weights_x0 = self.weights_x0[i].reshape(1, self.layer_sizes[i+1], self.layer_sizes[i])
      
      # finding layer output
      self.layers.append(self.lorentz(np.sum(batch_input * batch_weights_x0, axis=2), self.bias[i]))

    # resizing
    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.layer_sizes[-2])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.layer_sizes[-1], self.hidden_layer_sizes[-2])
    
    self.output = self.lorentz(np.sum(batch_input * batch_weights_x0, axis=2), self.bias[-1])

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences

    #output cost

    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])

    delta = [(2*(self.y - self.output)*self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2))).reshape(self.no_samples, self.output_size, 1)]

    # second to last layer delta

    if len(self.hidden_layer_sizes) > 1:
      batch_weights_x0 = self.weights_x0[-1].reshape(1, self.output_size, self.hidden_layer_sizes[-1])
        
      weights_T = np.asarray([mat.T for mat in batch_weights_x0])

      weight_delta = np.asarray([np.matmul(mat, delta) for mat, delta in zip(weights_T, delta[0])])

      batch_input = self.layers[-2].reshape(self.no_samples, 1, self.hidden_layer_sizes[-2])
      batch_weights_x0 = self.weights_x0[-2].reshape(1, self.hidden_layer_sizes[-1], self.hidden_layer_sizes[-2])

      deriv_mat = self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2)).reshape(self.no_samples, self.hidden_layer_sizes[-1], 1)

      delta.insert(0, weight_delta * deriv_mat)

      # deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
      # deriv_mat_T = np.asarray([mat.T for mat in deriv_mat])

      # delta.insert(0, np.asarray([np.matmul(mat, delta) for mat, delta in zip(deriv_mat_T, delta[0])]))
      # inserting new deltas at front of list
      for i in range(len(self.hidden_layer_sizes)-2, 0, -1):
        batch_weights_x0 = self.weights_x0[i+1].reshape(1, self.hidden_layer_sizes[i+1], self.hidden_layer_sizes[i])
        
        weights_T = np.asarray([mat.T for mat in batch_weights_x0])

        weight_delta = np.asarray([np.matmul(mat, delta) for mat, delta in zip(weights_T, delta[0])])

        batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])
        batch_weights_x0 = self.weights_x0[i].reshape(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])

        deriv_mat = self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2)).reshape(self.no_samples, self.hidden_layer_sizes[i], 1)

        delta.insert(0, weight_delta * deriv_mat)

      # finding delta for first layer
      batch_weights_x0 = self.weights_x0[1].reshape(1, self.hidden_layer_sizes[1], self.hidden_layer_sizes[0])
        
      weights_T = np.asarray([mat.T for mat in batch_weights_x0])

      weight_delta = np.asarray([np.matmul(mat, delta) for mat, delta in zip(weights_T, delta[0])])

      batch_input = self.input.reshape(self.no_samples, 1, self.input_size)
      batch_weights_x0 = self.weights_x0[0].reshape(1, self.hidden_layer_sizes[0], self.input_size)

      deriv_mat = self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2)).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)

      delta.insert(0, weight_delta * deriv_mat)

    else:

      batch_weights_x0 = self.weights_x0[1].reshape(1, self.output_size, self.hidden_layer_sizes[0])
        
      weights_T = np.asarray([mat.T for mat in batch_weights_x0])

      weight_delta = np.asarray([np.matmul(mat, delta) for mat, delta in zip(weights_T, delta[0])])

      batch_input = self.input.reshape(self.no_samples, 1, self.input_size)
      batch_weights_x0 = self.weights_x0[0].reshape(1, self.hidden_layer_sizes[0], self.input_size)

      deriv_mat = self.lorentzDx(np.sum(batch_input * batch_weights_x0, axis=2)).reshape(self.no_samples, self.hidden_layer_sizes[0], 1)

      delta.insert(0, weight_delta * deriv_mat)

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing
    batch_input = self.input.reshape(self.no_samples, 1, self.input_size)

    grad_weights_x0 = [np.multiply(batch_input, delta[0])]
    
    # for 2nd and more hidden layer weights
    for i in range(1, len(self.hidden_layer_sizes)):
      batch_input = self.layers[i-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[i-1])

      grad_weights_x0.append(np.multiply(batch_input, delta[i]))

    # for output layer weights
    batch_input = self.layers[-1].reshape(self.no_samples, 1, self.hidden_layer_sizes[-1])

    grad_weights_x0.append(np.multiply(batch_input, delta[-1]))
    
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
    return (0.5*self.gamma)/(np.pi * (np.square(x - x0) + np.square(0.5*self.gamma)))

  def lorentzDx(self, x):
    # derivative of lorentz function with respect to x
    return -4*(x - x0) * (np.pi / self.gamma) * np.square(self.lorentz(x))
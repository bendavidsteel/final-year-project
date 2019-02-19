import numpy as np
import torch

class FullLorentzianNeuralNetwork:
  def __init__(self, x, y, layers, alpha, gamma):

    # hyperparameters
    self.hidden_layer_sizes = layers
    self.alpha = alpha
    self.gamma = gamma

    # ensuring inputs and targets are np arrays
    self.input = torch.Tensor(x)
    
    # targets
    self.y = torch.Tensor(y)

    # parameters
    self.no_samples = y.shape[0]
    self.output_size = y.shape[1]
    self.input_size = self.input.shape[1]

    self.weights_x0 = []
    
    # first hidden layer weights
    self.weights_x0.append(torch.randn(self.hidden_layer_sizes[0], self.input_size))

    # additional hidden layer weights
    for i in range(len(self.hidden_layer_sizes)-1):
      self.weights_x0.append(torch.randn(self.hidden_layer_sizes[i+1], self.hidden_layer_sizes[i]))
    
    # output layer weights
    self.weights_x0.append(torch.randn(self.output_size, self.hidden_layer_sizes[-1]))

    # self.weights_x0 = torch.Tensor(self.weights_x0)

    self.output = torch.zeros(self.y.shape)

  def feedForward(self):
    # performing feed forward step
    self.layers = []

    # resizing input and weights for batch processing
    batch_input = self.input.view(self.no_samples, 1, self.input_size)
    batch_weights_x0 = self.weights_x0[0].view(1, self.hidden_layer_sizes[0], self.input_size)
    
    # finding layer output
    self.layers.append(torch.sum(self.lorentz(batch_input, batch_weights_x0), 2))

    for i in range(1, len(self.hidden_layer_sizes)):
      # resizing
      batch_input = self.layers[i-1].view(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].view(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      
      # finding layer output
      self.layers.append(torch.sum(self.lorentz(batch_input, batch_weights_x0), 2))

    # resizing
    batch_input = self.layers[-1].view(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].view(1, self.output_size, self.hidden_layer_sizes[-1])
    
    self.output = torch.sum(self.lorentz(batch_input, batch_weights_x0), 2)

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences
    delta = [(2*(self.y - self.output)).view(self.no_samples, self.output_size, 1)]

    batch_input = self.layers[-1].view(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].view(1, self.output_size, self.hidden_layer_sizes[-1])

    deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
    new_delta = [mat.t().mm(delta) for mat, delta in zip(deriv_mat, delta[0])]

    delta.insert(0, torch.cat(new_delta).view(self.no_samples, self.hidden_layer_sizes[-1], 1))
    # inserting new deltas at front of list
    for i in range(len(self.hidden_layer_sizes)-1, 0, -1):
      batch_input = self.layers[i-1].view(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].view(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])

      deriv_mat = self.lorentzDx(batch_input, batch_weights_x0)
      new_delta = [mat.t().mm(delta) for mat, delta in zip(deriv_mat, delta[0])]

      delta.insert(0, torch.cat(new_delta).view(self.no_samples, self.hidden_layer_sizes[i-1], 1))

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing
    batch_input = self.input.view(self.no_samples, 1, self.input_size)
    batch_weights_x0 = self.weights_x0[0].view(1, self.hidden_layer_sizes[0], self.input_size)
    
    deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)

    grad_weights_x0 = [torch.mul(deriv_mat, delta[0])]
    
    # for 2nd and more hidden layer weights
    for i in range(1, len(self.hidden_layer_sizes)):
      batch_input = self.layers[i-1].view(self.no_samples, 1, self.hidden_layer_sizes[i-1])
      batch_weights_x0 = self.weights_x0[i].view(1, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])
      
      deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)

      grad_weights_x0.append(torch.mul(deriv_mat, delta[i]))

    # for output layer weights
    batch_input = self.layers[-1].view(self.no_samples, 1, self.hidden_layer_sizes[-1])
    batch_weights_x0 = self.weights_x0[-1].view(1, self.output_size, self.hidden_layer_sizes[-1])
    
    deriv_mat = self.lorentzDx0(batch_input, batch_weights_x0)

    grad_weights_x0.append(torch.mul(deriv_mat, delta[-1]))
    
    grad_weights = [torch.div(torch.sum(weights, 0), self.no_samples) for weights in grad_weights_x0]

    # update the weights with the derivative of the loss function
    for i in range(len(self.weights_x0)):
        self.weights_x0[i] += self.alpha * grad_weights[i]
    

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
    return (0.5*self.gamma)/(np.pi * torch.add(torch.pow(torch.sub(x , x0), 2) , np.square(0.5*self.gamma)))

  def lorentzDx(self, x, x0):
    # derivative of lorentz function with respect to x
    return -4*torch.sub(x , x0)*(np.pi/self.gamma)*torch.pow(self.lorentz(x, x0), 2)

  def lorentzDx0(self, x, x0):
    # derivative of lorentz function with respect to x0
    return 4*torch.sub(x , x0)*(np.pi/self.gamma)*torch.pow(self.lorentz(x, x0), 2)

  def sigmoid(self, x):
    # sigmoid function
    return 1/(1 + torch.pow(np.e, -x))

  def derivSigmoid(self, x):
    # derivative of sigmoid function
    return np.exp(-x)/torch.pow(1 + np.exp(-x), 2)
# adapted from Alejandro Escontrela
# https://github.com/Alescontrela/Numpy-CNN

import numpy as np

# Needs extra light sources, normal weights with tunable activation function, (effectively bias weight)
# Convolutional
class ConvTuneLorentzNetwork:
  def __init__(self, x, y, full_layer_size = 128, alpha=0.001, gamma=1, num_classes = 10, img_dim = 28, img_depth = 1, f = 5, num_filt1 = 8, num_filt2 = 8, batch_size = 32, num_epochs = 2):

    # hyperparameters
    self.alpha = alpha
    self.gamma = gamma
    self.full_layer_size = full_layer_size

    # training data
    self.no_samples = 50000
    self.input = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    self.targets = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)

    # normalising
    self.input -= int(np.mean(self.input))
    self.input /= int(np.std(self.input))

    # partitioning data
    # SPLIT LATER ON INTO TRAINING AND VALIDATION
    train_data = np.hstack((self.input , self.targets))
    
    # shuffling dataset
    np.random.shuffle(train_data)

    ## initializing kernels
    f1, f2, w3, w4 = , , , 
    kernel1 = initializeFilter((num_filt1, img_depth, f, f))
    kernel2 = initializeFilter((num_filt2, num_filt1, f, f))

    num_conv_layers = 2
    self.flat_size = ((img_dim - num_conv_layers*(f - 1))**2) * num_filt2

    w3 = initializeWeight((full_layer_size, 800))
    w4 = initializeWeight((10, 128))


    self.bias = []

    for i in range(1, len(self.layer_sizes)):
      self.bias.append(np.random.rand(self.layer_sizes[i]) - 0.5)

    self.bias = np.asarray(self.bias)

    self.output = np.zeros(self.y.shape)

  def feedForward(self):
    # performing feed forward step
    self.layers_z = []
    self.layers_a = []

    # resizing input and weights for batch processing
    batch_input = self.input.reshape(self.no_samples, 1, self.layer_sizes[0])
    self.layers_a.append(batch_input)

    for i in range(0, len(self.weights)-1):
      # resizing
      batch_input = self.layers_a[-1]
      batch_weights = self.weights[i].reshape(1, self.layer_sizes[i+1], self.layer_sizes[i])
      
      # finding layer output
      z = np.sum(batch_input * batch_weights, axis=2)
      self.layers_z.append(z)

      batch_bias = self.bias[i].reshape(1, self.layer_sizes[i+1])
      self.layers_a.append(self.lorentz(z, batch_bias).reshape(self.no_samples, 1, self.layer_sizes[i+1]))

    # resizing
    batch_input = self.layers_a[-1]
    batch_weights = self.weights[-1].reshape(1, self.layer_sizes[-1], self.layer_sizes[-2])
    
    # finding layer output
    z = np.sum(batch_input * batch_weights, axis=2)
    self.layers_z.append(z)

    batch_bias = self.bias[-1].reshape(1, self.layer_sizes[-1])
    self.output = z + batch_bias

  def backProp(self):
    # application of the chain rule to find derivative of the loss function with respect to weights_x0 and weights_gamma
    # using cost function squared differences

    #output cost
    delta = [2*(self.y - self.output)]
    deriv = []

    # second to last layer delta

    new_delta = np.asarray([np.matmul(self.weights[-1].T, mat) for mat in delta[0]])

    delta.insert(0, new_delta)

    for i in range(len(self.weights)-2, 0, -1):
        deriv_mat = self.lorentzDx(self.layers_z[i], self.bias[i])
        deriv.insert(0, deriv_mat)

        delta_lorentz = deriv_mat * delta[0]

        new_delta = np.asarray([np.matmul(self.weights[i].T, mat) for mat in delta_lorentz])

        delta.insert(0, new_delta)

    deriv.insert(0, self.lorentzDx(self.layers_z[0], self.bias[0]))

    # finding the derivative with respect to weights
    # resizing input and weights for batch processing

    grad_weights = []
    grad_bias = []

    for i in range(0, len(delta)-1):
      delta_mat = delta[i].reshape(self.no_samples, self.layer_sizes[i+1], 1)
      deriv_mat = deriv[i].reshape(self.no_samples, self.layer_sizes[i+1], 1)

      grad_weights.append(delta_mat * deriv_mat * self.layers_a[i])
      grad_bias.append(delta[i] * self.lorentzDx0(self.layers_z[i], self.bias[i]))

    delta_mat = delta[-1].reshape(self.no_samples, self.layer_sizes[-1], 1)

    grad_weights.append(delta_mat * self.layers_a[-1])
    grad_bias.append(delta[-1])
    
    # average out contributions
    grad_weights_ave = [(np.sum(weights, axis=0) / self.no_samples) for weights in grad_weights]
    grad_bias_ave = [(np.sum(weights, axis=0) / self.no_samples) for weights in grad_bias]

    # update the weights with the derivative of the loss function
    self.weights += self.alpha * np.asarray(grad_weights_ave)
    self.bias += self.alpha * np.asarray(grad_bias_ave)
    

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

  def lorentzDx(self, x, x0):
    # derivative of lorentz function with respect to x
    return -4*(x - x0) * (np.pi / self.gamma) * np.square(self.lorentz(x, x0))

  def lorentzDx0(self, x, x0):
    # derivative of lorentz function with respect to x0
    return 4*(x - x0) * (np.pi / self.gamma) * np.square(self.lorentz(x, x0))
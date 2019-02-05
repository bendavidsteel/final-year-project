# adapted from Alejandro Escontrela
# https://github.com/Alescontrela/Numpy-CNN

import numpy as np
import gzip

# Needs extra light sources, normal weights with tunable activation function, (effectively bias weight)
# Convolutional
class ConvTuneLorentzNetwork:
  def __init__(self, x, y, full_layer_size = 128, alpha=0.001, gamma=1, no_classes = 10, img_dim = 28, img_depth = 1, f = 5, num_filt1 = 8, num_filt2 = 8, batch_size = 32, no_epochs = 2):

    # hyperparameters
    self.alpha = alpha
    self.gamma = gamma
    self.full_layer_size = full_layer_size
    self.no_classes = no_classes
    self.no_epochs = no_epochs
    self.batch_size = batch_size

    # training data
    self.no_samples = 50000
    self.input = self.extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    self.targets = self.extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)

    # normalising
    self.input -= int(np.mean(self.input))
    self.input /= int(np.std(self.input))

    # partitioning data
    # SPLIT LATER ON INTO TRAINING AND VALIDATION
    self.train_data = np.hstack((self.input , self.targets))
    
    # shuffling dataset
    np.random.shuffle(self.train_data)

    # initializing kernels
    self.kernel1 = self.initializeFilter((num_filt1, img_depth, f, f))
    self.kernel2 = self.initializeFilter((num_filt2, num_filt1, f, f))

    num_conv_layers = 2
    self.flat_size = ((img_dim - num_conv_layers*(f - 1))**2) * num_filt2

    self.weights3 = self.initializeWeight((self.full_layer_size, self.flat_size))
    self.weights4 = self.initializeWeight((self.no_classes, self.full_layer_size))

    self.bias1 = np.zeros((num_filt1, 1))
    self.bias2 = np.zeros((num_filt2, 1))
    self.bias3 = np.zeros((self.full_layer_size, 1))
    self.bias4 = np.zeros((self.no_classes, 1))

    # self.output = np.zeros(self.y.shape)

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
    

  def train(self):
    # train
    for epoch in range(self.no_epochs):
        np.random.shuffle(self.train_data)
        # TODO try not to use shape
        batches = [self.train_data[k:k + self.batch_size] for k in range(0, self.train_data.shape[0], self.batch_size)]

        # t = tqdm(batches)
        for i in range(len(batches)):
            x, batch = batches[i]
            # params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
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

  def initializeFilter(self, size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

  def initializeWeight(self, size):
    return np.random.standard_normal(size=size) * 0.01

  def extract_data(self, filename, num_images, IMAGE_WIDTH):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m 
    is the number of training examples.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

  def extract_labels(self, filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
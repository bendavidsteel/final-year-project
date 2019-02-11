'''
Description: Utility methods for a Convolutional Neural Network

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from CNN.forward import *
import numpy as np
import gzip

#####################################################
################## Utility Methods ##################
#####################################################
        
def extract_data(filename, num_images, IMAGE_WIDTH):
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

def extract_labels(filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.abs(np.random.normal(loc = 0, scale = stddev, size = size))

def initializeWeight(size):
    return np.abs(np.random.standard_normal(size=size) * 0.01)

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs    

def predict(image, label, params, config):
    '''
    Make predictions with trained filters/weights. 
    '''
    
    [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params 
    [num_filt1, num_filt2, num_filt3, conv_s, pool_f, pool_s, gamma] = config
    
    conv1 = convolution(image, f1, s = conv_s) # convolution operation
    nonlin1 = lorentz(conv1, b1.reshape(num_filt1, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    conv2 = convolution(nonlin1, f2, s = conv_s) # second convolution operation
    nonlin2 = lorentz(conv2, b2.reshape(num_filt2, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    conv3 = convolution(nonlin2, f3, s = pool_s) # using convolution with higher stride as pooling layer
    pooled = lorentz(conv3, b3.reshape(num_filt3, 1, 1), gamma)
    
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    z = w4.dot(fc) # first dense layer
    a = lorentz(z, b4, gamma) # pass through Lorentzian non-linearity
    
    out = w5.dot(a) + b5 # second dense layer

    out /= np.sum(out)
    
    # not using softmax as exponential cannot be implemented in optics
    
    return np.argmax(out), np.max(out)
    
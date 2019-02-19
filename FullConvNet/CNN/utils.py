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
    # scale by inverse square root of size of previous layer
    var = np.sqrt(6)/np.sqrt(size[0] + size[1])
    return np.abs(np.random.uniform(low = -var, high = var, size = size))

def initializeBias(size):
    var = np.sqrt(6)/np.sqrt(size[0])
    return np.random.uniform(low = -var, high = var, size = size)

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs    

def predict(image, label, params, config):
    '''
    Make predictions with trained filters/weights. 
    '''
    
    [f1, f2, w3, w4, w5] = params 
    [num_filt1, num_filt2, conv_s, gamma] = config
    
    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolutionLorentz(image, f1, gamma, s = conv_s) # convolution operation
    
    conv2 = convolutionLorentz(conv1, f2, gamma, s = conv_s) # second convolution operation
    
    (nf2, dim2, _) = conv2.shape
    fc = conv2.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    a1 = lorentz(fc.T, w3, gamma)
    z1 = np.sum(a1, axis=1).reshape((-1,1))

    a2 = lorentz(z1.T, w4, gamma)
    z2 = np.sum(a2, axis=1).reshape((-1,1))
    
    a3 = lorentz(z2.T, w5, gamma)
    out = np.sum(a3, axis=1).reshape((-1,1))

    out /= np.sum(out)
    
    # not using softmax as exponential cannot be implemented in optics
    
    return np.argmax(out), np.max(out)
    
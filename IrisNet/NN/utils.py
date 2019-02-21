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

def extract_semeion_digit_dataset():
	'''
	Extract semeion handwritten digit dataset
	'''
	with open('semeion.data', 'r') as f:

		digits = []
		labels = []

		for line in f:
			data = line.split(' ')

			digit_str = data[:256]
			label_str = data[256:266]
			
			digit = np.asarray([float(x) for x in digit_str])
			label_arr = np.asarray([int(x) for x in label_str])

			digits.append(digit)
			
			label = 0

			for i in range(len(label_arr)):
				if label_arr[i] == 1:
					label = i

			labels.append(label)

	return np.asarray(digits), np.asarray(labels).reshape((-1,1))

def semeion_training_set():
	digits, labels = extract_semeion_digit_dataset()
	# slice indexing to make up for non-random distribution of digits
	return np.concatenate((digits[1::5], digits[2::5], digits[3::5], digits[4::5])), np.concatenate((labels[1::5], labels[2::5], labels[3::5], labels[4::5]))

def semeion_validation_set():
	digits, labels = extract_semeion_digit_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return digits[::5], labels[::5]

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

def predict(image, label, params, gamma):
    '''
    Make predictions with trained filters/weights. 
    '''
    
    [f1, f2, w3, w4, b1, b2, b3, b4] = params 
    
    conv1 = convolution(image, f1) # convolution operation
    nonlin1 = lorentz(conv1, b1.reshape(-1, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    conv2 = convolution(nonlin1, f2) # second convolution operation
    pooled = lorentz(conv2, b2.reshape(-1, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    z = w3.dot(fc) # first dense layer
    a = lorentz(z, b3, gamma) # pass through Lorentzian non-linearity
    
    out = w4.dot(a) + b4 # second dense layer

    out /= np.sum(out)
    
    # not using softmax as exponential cannot be implemented in optics
    
    return np.argmax(out), np.max(out)
    
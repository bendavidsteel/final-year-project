'''
Description: Utility methods for a Convolutional Neural Network

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018

Altered by: Ben Steel
Date: 08/03/19
'''
from NN.forward import *
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

def extract_iris_dataset():
	'''
	Extract iris dataset
	'''
	with open('iris.data', 'r') as f:

		stats = []
		labels = []

		for line in f:
			data = line.split(',')

			stat_str = data[:4]
			label_str = data[4]
			
			stat = np.asarray([float(x) for x in stat_str])

			stats.append(stat)
			
			label = 0

			if 'Iris-setosa' in label_str:
				label = 0
			elif 'Iris-versicolor' in label_str:
				label = 1
			elif 'Iris-virginica' in label_str:
				label = 2

			labels.append(label)

	return np.asarray(stats), np.asarray(labels).reshape((-1,1))

def iris_training_set():
	stats, labels = extract_iris_dataset()
	# slice indexing to make up for non-random distribution of digits
	return np.concatenate((stats[::5], stats[1::5], stats[2::5])), np.concatenate((labels[::5], labels[1::5], labels[2::5]))

def iris_validation_set():
	stats, labels = extract_iris_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return stats[3::5], labels[3::5]

def iris_testing_set():
	stats, labels = extract_iris_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return stats[4::5], labels[4::5]

def extract_heart_dataset():
	'''
	Extract heart dataset
	'''
	with open('heart_shuffled.csv', 'r') as f:

		stats = []
		labels = []

		first = True

		for line in f:

			if first:
				first_line = line
				first = False
				continue

			data = line.split(',')

			stat_str = data[:13]
			label_str = data[13]
			
			stat = np.asarray([float(x) for x in stat_str])

			stats.append(stat)
			
			label = float(label_str)

			labels.append(label)

	return np.asarray(stats), np.asarray(labels).reshape((-1,1))

def heart_training_set():
	stats, labels = extract_heart_dataset()
	# slice indexing to make up for non-random distribution of digits
	return np.concatenate((stats[::5], stats[1::5], stats[2::5])), np.concatenate((labels[::5], labels[1::5], labels[2::5]))

def heart_validation_set():
	stats, labels = extract_heart_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return stats[3::5], labels[3::5]

def heart_testing_set():
	stats, labels = extract_heart_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return stats[4::5], labels[4::5]

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.abs(np.random.normal(loc = 0, scale = stddev, size = size))

def initializeWeight(size):
    # scale by inverse square root of size of previous layer
    var = np.sqrt(6)/np.sqrt(size[0] + size[1])
    return np.random.uniform(low = -var, high = var, size = size) + 1j*np.random.uniform(low = -var, high = var, size = size)

def initializeBias(size, bias_init):
	var = bias_init
	# return np.random.uniform(low = 0, high = var, size = size)
	return np.random.uniform(low = -var, high = var, size = size) + 1j*np.random.uniform(low = -var, high = var, size = size)

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs 

def norm_stack_shuffle(x, y_dash):
	x -= np.mean(x, axis=0)
	x /= np.std(x, axis=0)
	data = np.hstack((x,y_dash))
    
	np.random.shuffle(data)

	return data

def predict(data, label, params, gamma):
	'''
	Make predictions with trained filters/weights. 
	'''
	[w1, w2, w3, b1, b2, b3] = params

	if type(gamma) is not list:
		gamma = [gamma, gamma]

	z1 = w1.dot(data) # convolution operation
	a1 = nonlinComplex(z1.reshape(-1,1), b1, gamma[0]) # pass through Lorentzian non-linearity
    
	z2 = w2.dot(a1) # second convolution operation
	a2 = nonlinComplex(z2.reshape(-1,1), b2, gamma[1]) # pass through Lorentzian non-linearity

	out = w3.dot(a2) + b3.reshape(-1,1)

	measured = np.abs(out)

	softout = softmax(measured)
    
    # not using softmax as exponential cannot be implemented in optics
    
	return np.argmax(softout), np.max(softout)
    
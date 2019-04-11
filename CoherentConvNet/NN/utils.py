'''
Description: Utility methods for a Convolutional Neural Network

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from NN.forward import *
import numpy as np
import gzip
import pickle

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
	with open('semeion_shuffled.data', 'r') as f:

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
	return np.concatenate((digits[::5], digits[1::5], digits[2::5])), np.concatenate((labels[::5], labels[1::5], labels[2::5]))

def semeion_validation_set():
	digits, labels = extract_semeion_digit_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return digits[3::5], labels[3::5]

def semeion_testing_set():
	digits, labels = extract_semeion_digit_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return digits[4::5], labels[4::5]

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

def extract_shapes_dataset():
	'''
	Extract shapes dataset
	'''
	save_path = "shapes14.pkl"

	with open(save_path, 'rb') as f:
		shapes, labels = pickle.load(f)

	return shapes.reshape(-1, 14*14), labels.reshape(-1,1)

def shapes_training_set():
	shapes, labels = extract_shapes_dataset()
	# slice indexing to make up for non-random distribution of digits
	return np.concatenate((shapes[::5], shapes[1::5], shapes[2::5])), np.concatenate((labels[::5], labels[1::5], labels[2::5]))

def shapes_validation_set():
	shapes, labels = extract_shapes_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return shapes[3::5], labels[3::5]

def shapes_testing_set():
	shapes, labels = extract_shapes_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return shapes[4::5], labels[4::5]

def extract_fourshapes_dataset():
	'''
	Extract shapes dataset
	'''
	save_path = "fourshapes20.pkl"

	with open(save_path, 'rb') as f:
		shapes = pickle.load(f)

	return shapes[:5000, :400].astype(float), shapes[:5000, 400].reshape(-1,1).astype(float)

def fourshapes_training_set():
	shapes, labels = extract_fourshapes_dataset()
	# slice indexing to make up for non-random distribution of digits
	return np.concatenate((shapes[::5], shapes[1::5], shapes[2::5])), np.concatenate((labels[::5], labels[1::5], labels[2::5]))

def fourshapes_validation_set():
	shapes, labels = extract_fourshapes_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return shapes[3::5], labels[3::5]

def fourshapes_testing_set():
	shapes, labels = extract_fourshapes_dataset()
	# slice indexing to make up for non-random distribution of digits
	# splitting 80/20
	return shapes[4::5], labels[4::5]

def initializeFilter(size, scale = 1.0):
    var = scale/np.sqrt(np.prod(size))
    return np.random.uniform(low = -var, high = var, size = size) + 1j*np.random.uniform(low = -var, high = var, size = size)

def initializeWeight(size):
    # scale by inverse square root of size of previous layer
    var = np.sqrt(6)/np.sqrt(size[0] + size[1])
    return np.random.uniform(low = -var, high = var, size = size) + 1j*np.random.uniform(low = -var, high = var, size = size)

def initializeBias(size):
	var = 4
	# return np.random.uniform(low = 0, high = var, size = size)
	return np.random.uniform(low = -var, high = var, size = size) + 1j*np.random.uniform(low = -var, high = var, size = size)

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs 

def norm_stack_shuffle(x, y_dash, by_column=True):

	if by_column:
		x -= np.mean(x, axis=0)
		x /= np.std(x, axis=0)
	else:
		x -= np.mean(x)
		x /= np.std(x)

	data = np.hstack((x,y_dash))
    
	np.random.shuffle(data)

	return data

def predict(image, label, params, gamma):
	'''
	Make predictions with trained filters/weights. 
	'''
	[f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params

	conv1 = convolutionComplex(image, f1) # convolution operation
	nonlin1 = nonlinComplex(conv1, b1.reshape(-1, 1, 1), gamma) # pass through Lorentzian non-linearity

	conv2 = convolutionComplex(nonlin1, f2) # second convolution operation
	nonlin2 = nonlinComplex(conv2, b2.reshape(-1, 1, 1), gamma) # pass through Lorentzian non-linearity

	conv3 = convolutionComplex(nonlin2, f3, s=2) # second convolution operation
	pooled = nonlinComplex(conv3, b3.reshape(-1, 1, 1), gamma) # pass through Lorentzian non-linearity

	(nf2, dim2, _) = pooled.shape
	fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer

	z1 = w4.dot(fc) # second convolution operation
	a1 = nonlinComplex(z1, b4.reshape(-1,1), gamma) # pass through Lorentzian non-linearity

	# z2 = np.matmul(w4, a1) # first dense layer
	# a2 = lorentz(z2, b4.reshape(1,-1,1), gamma) # pass through Lorentzian non-linearity

	# out = np.matmul(w5, a2) + b5.reshape(1,-1,1) # second dense layer

	out = w5.dot(a1) + b5.reshape(-1,1)

	measured = np.real(out)**2 + np.imag(out)**2

	prob = softmax(measured)
	# not using softmax as exponential cannot be implemented in optics

	return np.argmax(prob), np.max(prob)

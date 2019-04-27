'''Author : Ben Steel
Date : 15/03/19'''

import numpy as np 
import matplotlib.pyplot as plt 
import pickle 

def plot_three_shapes():
	save_path = "shapes14.pkl"

	shapes, labels = pickle.load(open(save_path, 'rb'))

	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,6))

	for i in range(3):
		index = 0
		while labels[index] != i:
			index += 1
		else:
			axes[i].imshow(shapes[index], cmap='Greys')
			axes[i].set_aspect('equal')

			for tic in axes[i].xaxis.get_major_ticks():
				tic.tick1On = tic.tick2On = False
				tic.label1On = tic.label2On = False

			for tic in axes[i].yaxis.get_major_ticks():
				tic.tick1On = tic.tick2On = False
				tic.label1On = tic.label2On = False

	plt.show()


def plot_mnist():
	save_path = 'mnist.pkl'

	with open(save_path, 'rb') as f:
		data = pickle.load(f)

	images = data['training_images']
	labels = data['training_labels']

	fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16,6))

	for i in range(5):
		index = 0
		while labels[index] != i:
			index += 1
		else:
			axes[0,i].imshow(images[index].reshape(28, 28), cmap='Greys')
			axes[0,i].set_aspect('equal')

			for tic in axes[0,i].xaxis.get_major_ticks():
				tic.tick1On = tic.tick2On = False
				tic.label1On = tic.label2On = False

			for tic in axes[0,i].yaxis.get_major_ticks():
				tic.tick1On = tic.tick2On = False
				tic.label1On = tic.label2On = False

	for i in range(5, 10):
		index = 0
		while labels[index] != i:
			index += 1
		else:
			axes[1,i-5].imshow(images[index].reshape(28, 28), cmap='Greys')
			axes[1,i-5].set_aspect('equal')

			for tic in axes[1,i-5].xaxis.get_major_ticks():
				tic.tick1On = tic.tick2On = False
				tic.label1On = tic.label2On = False

			for tic in axes[1,i-5].yaxis.get_major_ticks():
				tic.tick1On = tic.tick2On = False
				tic.label1On = tic.label2On = False

	plt.show()

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

plot_three_shapes()
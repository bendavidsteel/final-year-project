'''Author: Ben Steel
Date: 11/02/19'''

import matplotlib.pyplot as plt
import numpy as np 
import pickle

if __name__ == '__main__':

	num_gammas = 10
	iters = 5

	fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17,7))

	eval_save_path = "layer_gamma_accuracy_1616_0.05_15_heart_ne5000_bias10b.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[x_g, y_g, g_a, e_n] = to_save

	x = y_g[:num_gammas]
	y = x

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = g_a[i*num_gammas + j]

	axes[0,0].contourf(x/2, y/2, z, 20, vmin=30, vmax=100)
	axes[0,0].set_aspect('equal')
	axes[0,0].set_xlabel(r'$\kappa$ for First Layer')
	axes[0,0].set_ylabel(r'$\kappa$ for Second Layer')
	axes[0,0].set_title("[16,16]")

	for i in range(num_gammas):
		for j in range(num_gammas):
			z[j,i] = e_n[i*num_gammas + j]
			# z[j,i] = g_a[i*num_gammas + j]

	axes[1,0].contourf(x/2, y/2, z, 20, vmin=0, vmax=5000, cmap='plasma')
	axes[1,0].set_aspect('equal')
	axes[1,0].set_xlabel(r'$\kappa$ for First Layer')
	axes[1,0].set_ylabel(r'$\kappa$ for Second Layer')
	# axes[1,0].set_title("[16,16]")

	eval_save_path = "layer_gamma_accuracy_1632_0.05_15_heart_ne5000_bias10b.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[x_g, y_g, g_a, e_n] = to_save

	x = y_g[:num_gammas]
	y = x

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = g_a[i*num_gammas + j]

	axes[0,1].contourf(x/2, y/2, z, 20, vmin=30, vmax=100)
	axes[0,1].set_aspect('equal')
	axes[0,1].set_xlabel(r'$\kappa$ for First Layer')
	axes[0,1].set_ylabel(r'$\kappa$ for Second Layer')
	axes[0,1].set_title("[16,32]")

	for i in range(num_gammas):
		for j in range(num_gammas):
			z[j,i] = e_n[i*num_gammas + j]
			# z[j,i] = g_a[i*num_gammas + j]

	axes[1,1].contourf(x/2, y/2, z, 20, vmin=0, vmax=5000, cmap='plasma')
	axes[1,1].set_aspect('equal')
	axes[1,1].set_xlabel(r'$\kappa$ for First Layer')
	axes[1,1].set_ylabel(r'$\kappa$ for Second Layer')
	# axes[0,1].set_title("[16,64]")

	eval_save_path = "layer_gamma_accuracy_3216_0.05_15_heart_ne5000_bias10b.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[x_g, y_g, g_a, e_n] = to_save

	x = y_g[:num_gammas]
	y = x

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = g_a[i*num_gammas + j]

	im1 = axes[0,2].contourf(x/2, y/2, z, 20, vmin=30, vmax=100)
	axes[0,2].set_aspect('equal')
	axes[0,2].set_xlabel(r'$\kappa$ for First Layer')
	axes[0,2].set_ylabel(r'$\kappa$ for Second Layer')
	axes[0,2].set_title("[32,16]")

	for i in range(num_gammas):
		for j in range(num_gammas):
			z[j,i] = e_n[i*num_gammas + j]
			# z[j,i] = g_a[i*num_gammas + j]

	axes[1,2].contourf(x/2, y/2, z, 20, vmin=0, vmax=5000, cmap='plasma')
	axes[1,2].set_aspect('equal')
	axes[1,2].set_xlabel(r'$\kappa$ for First Layer')
	axes[1,2].set_ylabel(r'$\kappa$ for Second Layer')
	# axes[1,2].set_title("[64,16]")

	eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_bias10b.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[x_g, y_g, g_a, e_n] = to_save

	x = y_g[:num_gammas]
	y = x

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = g_a[i*num_gammas + j]

	axes[0,3].contourf(x/2, y/2, z, 20, vmin=30, vmax=100)
	axes[0,3].set_aspect('equal')
	axes[0,3].set_xlabel(r'$\kappa$ for First Layer')
	axes[0,3].set_ylabel(r'$\kappa$ for Second Layer')
	axes[0,3].set_title("[32,32]")

	for i in range(num_gammas):
		for j in range(num_gammas):
			z[j,i] = e_n[i*num_gammas + j]
			# z[j,i] = g_a[i*num_gammas + j]

	im2 = axes[1,3].contourf(x/2, y/2, z, 20, vmin=0, vmax=5000, cmap='plasma')
	axes[1,3].set_aspect('equal')
	axes[1,3].set_xlabel(r'$\kappa$ for First Layer')
	axes[1,3].set_ylabel(r'$\kappa$ for Second Layer')
	# axes[1,1].set_title("[64,64]")

	# plt.tight_layout()

	cbar1 = fig.colorbar(im1, ax=[axes[0,0], axes[0,1], axes[0,2], axes[0,3]])
	cbar1.set_label("Accuracy")

	cbar2 = fig.colorbar(im2, ax=[axes[1,0], axes[1,1], axes[1,2], axes[1,3]])
	cbar2.set_label("Epochs to Best Performance")
	# cbar.set_label("Epochs to Best Performance")

	plt.show()
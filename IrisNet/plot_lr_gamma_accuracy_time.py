import matplotlib.pyplot as plt
import numpy as np 
import pickle

if __name__ == '__main__':

	num_gammas = 20
	iters = 5

	fig, axes = plt.subplots(nrows=2, ncols=2)

	eval_save_path = "layer_gamma_accuracy_1664_0.05_15.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[x_g, y_g, g_a, e_n] = to_save

	x = y_g[:num_gammas]
	y = x

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = g_a[i*num_gammas + j]

	axes[0,0].contourf(x, y, z, 20)
	axes[0,0].set_xlabel(r'$\kappa$ for First Layer')
	axes[0,0].set_ylabel(r'$\kappa$ for Second Layer')
	axes[0,0].set_title("LearningRate = 0.01, MaxEpochs = 2000")

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = e_n[i*num_gammas + j]

	axes[1,0].contourf(x, y, z, 20)
	axes[1,0].set_xlabel(r'$\kappa$ for First Layer')
	axes[1,0].set_ylabel(r'$\kappa$ for Second Layer')
	# axes[1,0].set_title("[16,64]")

	eval_save_path = "layer_gamma_accuracy_1664_0.05_15_heart_ne5000_lr005.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[x_g, y_g, g_a, e_n] = to_save

	x = y_g[:num_gammas]
	y = x

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = g_a[i*num_gammas + j]

	axes[0,1].contourf(x, y, z, 20)
	axes[0,1].set_xlabel(r'$\kappa$ for First Layer')
	axes[0,1].set_ylabel(r'$\kappa$ for Second Layer')
	axes[0,1].set_title("LearningRate = 0.05, MaxEpochs = 5000")

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = e_n[i*num_gammas + j]

	im = axes[1,1].contourf(x, y, z, 20)
	axes[1,1].set_xlabel(r'$\kappa$ for First Layer')
	axes[1,1].set_ylabel(r'$\kappa$ for Second Layer')
	# axes[1,1].set_title("[64,64]")

	plt.tight_layout()

	cbar = fig.colorbar(im, ax=axes.ravel().tolist())
	cbar.set_label("Accuracy")
	# cbar.set_label("Epochs to Best Performance")

	plt.show()
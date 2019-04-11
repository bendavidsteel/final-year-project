import matplotlib.pyplot as plt
import numpy as np 
import pickle

if __name__ == '__main__':

	# num_gammas = 20
	num_gammas = 10
	iters = 5

	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16,8))

	# eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_lr001.pkl"
	eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_bias1.pkl"
	# eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_bias10neg.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[x_g, y_g, g_a, e_n] = to_save

	x = y_g[:num_gammas]
	y = x

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = g_a[i*num_gammas + j]

	axes[0,0].contourf(x/2, y/2, z, 20, vmin=45, vmax=85)
	axes[0,0].set_aspect('equal')
	axes[0,0].set_xlabel(r'First layer $\kappa$')
	axes[0,0].set_ylabel(r'Second layer $\kappa$')
	axes[0,0].set_title(r'$x_{0} \in [-1, 1]$')

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = e_n[i*num_gammas + j]

	axes[1,0].contourf(x/2, y/2, z, 20, vmin=0, vmax=4000, cmap='plasma')
	axes[1,0].set_aspect('equal')
	axes[1,0].set_xlabel(r'First layer $\kappa$')
	axes[1,0].set_ylabel(r'Second layer $\kappa$')
	# axes[1,0].set_title("[16,64]")

	# eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_lr01.pkl"
	eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_bias5b.pkl"
	# eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_bias10.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[x_g, y_g, g_a, e_n] = to_save

	x = y_g[:num_gammas]
	y = x

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = g_a[i*num_gammas + j]

	axes[0,1].contourf(x/2, y/2, z, 20, vmin=45, vmax=85)
	axes[0,1].set_aspect('equal')
	axes[0,1].set_xlabel(r'First layer $\kappa$')
	axes[0,1].set_ylabel(r'Second layer $\kappa$')
	axes[0,1].set_title(r'$x_{0} \in [-5, 5]$')

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = e_n[i*num_gammas + j]

	im2 = axes[1,1].contourf(x/2, y/2, z, 20, vmin=0, vmax=4000, cmap='plasma')
	axes[1,1].set_aspect('equal')
	axes[1,1].set_xlabel(r'First layer $\kappa$')
	axes[1,1].set_ylabel(r'Second layer $\kappa$')
	# axes[1,1].set_title("[64,64]")

	# eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_lr1.pkl"
	eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_bias10.pkl"
	# eval_save_path = "layer_gamma_accuracy_3232_0.05_15_heart_ne5000_bias10pos.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[x_g, y_g, g_a, e_n] = to_save

	x = y_g[:num_gammas]
	y = x

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = g_a[i*num_gammas + j]

	im1 = axes[0,2].contourf(x/2, y/2, z, 20, vmin=45, vmax=85)
	axes[0,2].set_aspect('equal')
	axes[0,2].set_xlabel(r'First layer $\kappa$')
	axes[0,2].set_ylabel(r'Second layer $\kappa$')
	axes[0,2].set_title(r'$x_{0} \in [-10, 10]$')

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_gammas):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = e_n[i*num_gammas + j]

	axes[1,2].contourf(x/2, y/2, z, 20, vmin=0, vmax=4000, cmap='plasma')
	axes[1,2].set_aspect('equal')
	axes[1,2].set_xlabel(r'First layer $\kappa$')
	axes[1,2].set_ylabel(r'Second layer $\kappa$')
	# axes[1,2].set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8'])
	# axes[1,2].set_title("[64,64]")

	# plt.tight_layout()

	cbar1 = fig.colorbar(im1, ax=[axes[0,0], axes[0,1], axes[0,2]])
	cbar1.set_label("Accuracy")

	cbar2 = fig.colorbar(im2, ax=[axes[1,0], axes[1,1], axes[1,2]])
	cbar2.set_label("Epochs to Best Performance")

	# plt.tight_layout()
	plt.show()
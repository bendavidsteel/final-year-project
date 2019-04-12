import matplotlib.pyplot as plt
import numpy as np 
import pickle

if __name__ == '__main__':

	# num_gammas = 20
	num_gammas = 15
	num_inits = 15
	iters = 3

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

	eval_save_path = "gamma_vs_init_accuracy_coherent_3232_heart_ne5000.pkl"

	to_save = pickle.load(open(eval_save_path, 'rb')) 

	[gammas, inits, acc, epochs] = to_save

	z = np.zeros((num_gammas, num_gammas))

	for i in range(num_gammas):
		for j in range(num_inits):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = acc[i*num_gammas + j]

	im1 = axes[0].contourf(gammas[::num_gammas]/2, inits[:num_inits], z, 20)
	axes[0].set_xlabel(r'$\kappa$ Value')
	axes[0].set_ylabel('Initialisation Variance')
	# axes[0,0].set_title(r'$x_{0} \in [-1, 1]$')

	for i in range(num_gammas):
		for j in range(num_inits):
			# z[j,i] = e_n[i*num_gammas + j]
			z[j,i] = epochs[i*num_inits + j]

	im2 = axes[1].contourf(gammas[::num_gammas]/2, inits[:num_inits], z, 20, cmap='plasma')
	axes[1].set_xlabel(r'$\kappa$ Value')
	axes[1].set_ylabel('Initialisation Variance')
	# axes[0,0].set_title(r'$x_{0} \in [-1, 1]$')

	# plt.tight_layout()

	cbar1 = fig.colorbar(im1, ax=[axes[0]])
	cbar1.set_label("Accuracy")

	cbar2 = fig.colorbar(im2, ax=[axes[1]])
	cbar2.set_label("Epochs to Best Performance")

	# plt.tight_layout()
	plt.show()
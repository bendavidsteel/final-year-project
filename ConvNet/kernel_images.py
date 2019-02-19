import numpy as np
import matplotlib.pyplot as plt
import pickle

from CNN.forward import *
from SimpleDigitDataset import *

if __name__ == '__main__':

    save_path = 'adamGD_SoftmaxCross_2overpiGamma_Net128_4kernels_SimpleDigits_NLquartiledata'

    params, cost, layer_q5, layer_q25, layer_q50, layer_q75, layer_q95, final_layer = pickle.load(open(save_path, 'rb'))

    [f1, f2, w3, w4, b1, b2, b3, b4] = params 
    gamma = 2 / np.pi

    images, labels = generateDataset()

    images -= np.mean(images)
    images /= np.std(images)

    im1 = images[7].reshape((1,8,8))
    im2 = images[27].reshape((1,8,8))
    im3 = images[33].reshape((1,8,8))

    conv1 = convolution(im1, f1) # convolution operation
    nonlin1 = lorentz(conv1, b1.reshape(-1, 1, 1), gamma) # pass through Lorentzian non-linearity

    conv2 = convolution(im2, f1) # convolution operation
    nonlin2 = lorentz(conv2, b1.reshape(-1, 1, 1), gamma) # pass through Lorentzian non-linearity

    conv3 = convolution(im3, f1) # convolution operation
    nonlin3 = lorentz(conv3, b1.reshape(-1, 1, 1), gamma) # pass through Lorentzian non-linearity

    fig, axes = plt.subplots(3, 5)

    axes[0,0].imshow(im1[0], cmap='Greys')
    axes[0,0].set_title('Input')
    for i in range(4):
        axes[0,i+1].imshow(nonlin1[i], cmap='Greys')
        axes[0,i+1].set_title('Kernel '+str(i+1))

    axes[1,0].imshow(im2[0], cmap='Greys')
    for i in range(4):
        axes[1,i+1].imshow(nonlin2[i], cmap='Greys')

    axes[2,0].imshow(im3[0], cmap='Greys')
    for i in range(4):
        axes[2,i+1].imshow(nonlin3[i], cmap='Greys')

    [axi.set_xticks([]) for axi in axes.ravel()]
    [axi.set_yticks([]) for axi in axes.ravel()]
    plt.show()
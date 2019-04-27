'''Author: Ben Steel
Date: 15/03/19'''

import numpy as np
import matplotlib.pyplot as plt 
import CNN.utils as utils

if __name__ == '__main__':
    
    digits, labels = utils.extract_medium_digit_dataset()

    digits -= np.mean(digits)
    digits /= np.std(digits)

    im = []

    for i in range(10):
        for j in range(len(digits)):
            if labels[j] == i:
                im.append(digits[j].reshape((16,16)))
                break

    fig, axes = plt.subplots(2, 5)

    for i in range(5):
        axes[0,i].imshow(im[i], cmap='Greys')

    for i in range(5):
        axes[1,i].imshow(im[i+5], cmap='Greys')

    [axi.set_xticks([]) for axi in axes.ravel()]
    [axi.set_yticks([]) for axi in axes.ravel()]
    plt.show()
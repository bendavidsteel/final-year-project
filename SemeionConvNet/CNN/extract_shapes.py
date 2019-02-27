import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.measure import block_reduce
import pickle

if __name__ == "__main__":

    root = "shapes/"

    shape = "circles/"

    images = []
    labels = []

    for i in range(1, 101):
        img=mpimg.imread(root+shape+'drawing('+str(i)+').png')

        grey = rgb2gray(img)

        sub = block_reduce(grey, (2, 2))

        images.append(sub)
        labels.append(0)

    shape = "squares/"

    for i in range(1, 101):
        img=mpimg.imread(root+shape+'drawing('+str(i)+').png')

        grey = rgb2gray(img)

        sub = block_reduce(grey, (2, 2))

        images.append(sub)
        labels.append(1)

    shape = "triangles/"

    for i in range(1, 101):
        img=mpimg.imread(root+shape+'drawing('+str(i)+').png')

        grey = rgb2gray(img)

        sub = block_reduce(grey, (2, 2))

        images.append(sub)
        labels.append(2)

    images = np.asarray(images)
    labels = np.asarray(labels)

    to_save = [images, labels]
        
    save_path = "shapes14.pkl"
    
    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)
import numpy as np 
import pickle

f_s = 3

accuracies = []

for i in range(3):
    with open('p4b5g4/mnist_16f'+str(f_s)+'_32f'+str(f_s)+'_p4_1f512_g4_lr_01_bias10b_try'+str(i)+'.pkl', 'rb') as f:
        data = pickle.load(f)
        accuracies.append(data[4])

max_accuracy = max(accuracies)

lowest_error = 1 - max_accuracy

x=1
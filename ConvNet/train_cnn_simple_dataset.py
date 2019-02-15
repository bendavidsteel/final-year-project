'''
Description: Script to train the network and measure its performance on the test set.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from CNN.network_simple_dataset import *
from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_2overpiGamma_AlteredNet_SimpleDigits_NLdata'
    gamma = 2/np.pi

    cost = train(gamma = gamma, save_path = save_path)

    params, cost, layer_mean, layer_std, final_layer = pickle.load(open(save_path, 'rb'))
    [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params
    
    # Plot cost 
    plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()

    # Get test data
    X, y_dash = generateDataset()
    # Normalize the data
    X -= np.mean(X) # subtract mean
    X /= np.std(X) # divide by standard deviation
    test_data = np.hstack((X,y_dash))
    
    X = test_data[:,0:-1]
    X = X.reshape(len(test_data), 1, 8, 8)
    y = test_data[:,-1]

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    num_filt1 = num_filt2 = 5
    conv_s = 1

    params = [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5]
    config = [num_filt1, num_filt2, conv_s, gamma]

    for i in t:
        x = X[i]
        pred, prob = predict(x, y, params, config)
        digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    x = np.arange(10)
    digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x,digit_recall)
    plt.show()
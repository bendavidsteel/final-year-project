'''
Description: Script to train the network and measure its performance on the test set.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from CNN.network_shapes import *
from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_2overpiGamma_Net128_ShapesDataset'
    gamma = 2/np.pi

    cost = train(gamma = gamma, save_path = save_path, continue_training = False)

    params, cost, cost_val = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    # Plot cost 
    plt.plot(cost)
    plt.plot(np.linspace(0, len(cost), len(cost_val)), cost_val)
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()

    # Get test data
    X, y_dash = shapes_testing_set()
    # Normalize the data
    X -= np.mean(X) # subtract mean
    X /= np.std(X) # divide by standard deviation
    test_data = np.hstack((X,y_dash))
    
    X = test_data[:,0:-1]
    X = X.reshape(len(test_data), 1, 14, 14)
    y = test_data[:,-1]

    corr = 0
    digit_count = [0 for i in range(3)]
    digit_correct = [0 for i in range(3)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    for i in t:
        x = X[i]
        pred, prob = predict(x, y, params, gamma)
        digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    x = ["Circles", "Squares", "Triangles"]
    digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
    plt.xlabel('Shapes')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x,digit_recall)
    plt.show()
'''
Description: Script to train the network and measure its performance on the test set.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from NN.network import *
from NN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_2overpiGamma_IncoherentNet1616_heartDataset'
    gamma = 2/np.pi

    cost = train(gamma = gamma, save_path = save_path, continue_training = False)

    params, cost, cost_val, num_epochs = pickle.load(open(save_path, 'rb'))
    [w1, w2, w3, b1, b2, b3] = params
    
    # Plot cost 
    plt.plot(cost)
    plt.plot(np.linspace(0, len(cost), len(cost_val)), cost_val)
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.show()

    # Get test data
    X, y_dash = heart_testing_set()
    # Normalize the data
    test_data = norm_stack_shuffle(X,y_dash)

    num_classes = 2
    
    X = test_data[:,0:-1]
    y = test_data[:,-1]

    corr = 0
    iris_count = [0 for i in range(num_classes)]
    iris_correct = [0 for i in range(num_classes)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    params = [w1, w2, w3, b1, b2, b3]

    for i in t:
        x = X[i]
        pred, prob = predict(x, y, params, gamma)
        iris_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            iris_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    labels = ["No presence", "Presence"]
    iris_recall = [x/y for x,y in zip(iris_correct, iris_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(labels, iris_recall)
    plt.show()
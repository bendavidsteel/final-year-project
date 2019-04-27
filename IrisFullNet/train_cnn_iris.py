'''
Description: Script to train the network and measure its performance on the test set.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018

Altered by: Ben Steel
Date: 10/02/19
'''
from NN.network_iris import *
from NN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_2overpiGamma_FullNet1616_IrisDataset'
    gamma = 2/np.pi

    cost = train(gamma = gamma, save_path = save_path, continue_training = True)

    params, cost = pickle.load(open(save_path, 'rb'))
    [w1, w2, w3] = params
    
    # Plot cost 
    plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()

    # Get test data
    X, y_dash = iris_validation_set()
    # Normalize the data
    X -= np.mean(X) # subtract mean
    X /= np.std(X) # divide by standard deviation
    test_data = np.hstack((X,y_dash))
    
    X = test_data[:,0:-1]
    y = test_data[:,-1]

    corr = 0
    iris_count = [0 for i in range(3)]
    iris_correct = [0 for i in range(3)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    params = [w1, w2, w3]

    for i in t:
        x = X[i]
        pred, prob = predict(x, y, params, gamma)
        iris_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            iris_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    iris_recall = [x/y for x,y in zip(iris_correct, iris_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(labels, iris_recall)
    plt.show()
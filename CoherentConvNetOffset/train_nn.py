'''
Description: Script to train the network and measure its performance on the test set.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from NN.network_alt import *
from NN.utils import *

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import argparse
import matplotlib.pyplot as plt
import pickle
import warnings

warnings.simplefilter('error')

if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_1Gamma_bias1b_CoherentNet_shapesDataset_NLdata_offset10_try0.pkl'
    gamma = 1

    cost = train(gamma = gamma, save_path = save_path, continue_training = False, offset=10)

    with open(save_path, 'rb') as f:
        params, cost, cost_val, nl1_p, nl2_p, nl3_p, final_layer = pickle.load(f)

    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    # Plot cost 
    plt.plot(cost)
    plt.plot(np.linspace(0, len(cost), len(cost_val)), cost_val)
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.show()

    # Get test data
    X, y_dash = shapes_testing_set()
    # Normalize the data
    test_data = norm_stack_shuffle(X,y_dash, offset, by_column=False)

    num_classes = 3
    
    X = test_data[:,0:-1]
    X = X.reshape(len(test_data), 1, 14, 14)
    y = test_data[:,-1]

    corr = 0
    # iris_count = [0 for i in range(num_classes)]
    # iris_correct = [0 for i in range(num_classes)]
    test_actu = []
    test_pred = []
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    for i in t:
        x = X[i]
        pred, prob = predict(x, y, params, gamma)

        test_actu.append(int(y[i]))
        test_pred.append(pred)

        # iris_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
        #     iris_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    # labels = ["No presence", "Presence"]
    # iris_recall = [x/y for x,y in zip(iris_correct, iris_count)]
    # plt.xlabel('Digits')
    # plt.ylabel('Recall')
    # plt.title("Recall on Test Set")
    # plt.bar(labels, iris_recall)
    # plt.show()

    conf_mat = confusion_matrix(test_actu, test_pred)

    mat = conf_mat / conf_mat.astype(np.float).sum(axis=1)

    fig, ax = plt.subplots()

    ax.matshow(mat, cmap="gray_r") # imshow
    ax.xaxis.set_ticks_position('bottom')

    for i in range(num_classes):
        for j in range(num_classes):
            s = conf_mat[j,i]

            if mat[j,i] > 0.5:
                c = "white"
            else:
                c = "black"

            ax.text(i, j, str(s), color=c, va='center', ha='center')

    plt.title("Confusion Matrix")
    plt.xticks((0,1,2), ("Circles", "Squares", "Triangles"))
    plt.yticks((0,1,2), ("Circles", "Squares", "Triangles"))
    #plt.tight_layout()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.show()
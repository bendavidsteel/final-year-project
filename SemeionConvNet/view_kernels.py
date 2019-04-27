'''Author : Ben Steel
Date : 04/03/19'''

import numpy as np 
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from CNN.utils import *

save_path = 'adamGD_SoftmaxCross_4Gamma_bias10b_Net128_NLData_ShapesDataset_1000epochs_filter7_1.pkl'

gamma = 4

with open(save_path, 'rb') as f:
    data = pickle.load(f)

[params, final_layer] = data

[f1, f2, w3, w4, b1, b2, b3, b4] = params

(n_f, _, f, f) = f1.shape

# Get test data
X, y_dash = shapes_testing_set()
# Normalize the data
# Normalize the data
X -= np.mean(X) # subtract mean
X /= np.std(X) # divide by standard deviation
test_data = np.hstack((X,y_dash))

num_classes = 3

X = test_data[:,0:-1]
X = X.reshape(len(test_data), 1, 14, 14)
y = test_data[:,-1]

corr = 0
# iris_count = [0 for i in range(num_classes)]
# iris_correct = [0 for i in range(num_classes)]
test_actu = []
test_pred = []

params = [f1, f2, w3, w4, b1, b2, b3, b4]

for i in range(len(X)):
    x = X[i]
    pred, prob = predict(x, y, params, gamma)

    test_actu.append(int(y[i]))
    test_pred.append(pred)

    # iris_count[int(y[i])]+=1
    if pred==y[i]:
        corr+=1
    #     iris_correct[pred]+=1
    
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

fig = plt.figure(figsize=(16,4))

ax = plt.subplot(133)

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

for i in range(n_f):
    plot_num = i + 1 + (i//(n_f//2))*2
    plt.subplot(2,6,plot_num).imshow(f1[i,0,:,:], cmap='Greys')
    plt.subplot(2,6,plot_num).set_aspect('equal')
    plt.subplot(2,6,plot_num).set_title('Kernel ' + str(i + 1))

    for tic in plt.subplot(2,6,plot_num).xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

    for tic in plt.subplot(2,6,plot_num).yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

plt.show()
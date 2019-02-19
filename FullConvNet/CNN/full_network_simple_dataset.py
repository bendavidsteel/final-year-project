'''
Description: methods to set up and train the network's parameters.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''

'''
Adapted by Ben Steel
Date: 05/02/19
'''

from CNN.forward import *
from CNN.backward import *
from CNN.utils import *
from SimpleDigitDataset import *

import numpy as np
import pickle
from tqdm import tqdm
#####################################################
############### Building The Network ################
#####################################################

def conv(image, label, params, config):
    
    [f1, f2, w3, w4, w5] = params 
    [num_filt1, num_filt2, conv_s, gamma] = config
    
    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolutionLorentz(image, f1, gamma, s = conv_s) # convolution operation
    
    conv2 = convolutionLorentz(conv1, f2, gamma, s = conv_s) # second convolution operation
    
    (nf2, dim2, _) = conv2.shape
    fc = conv2.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    a1 = lorentz(fc.T, w3, gamma)
    z1 = np.sum(a1, axis=1).reshape((-1,1))

    a2 = lorentz(z1.T, w4, gamma)
    z2 = np.sum(a2, axis=1).reshape((-1,1))
    
    a3 = lorentz(z2.T, w5, gamma)
    out = np.sum(a3, axis=1).reshape((-1,1))

    probs = softmax(out) # predict class probabilities with the softmax activation function
    
    ################################################
    #################### Loss ######################
    ################################################
    
    loss = categoricalCrossEntropy(probs, label) # categorical cross-entropy loss
        
    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label # derivative of loss w.r.t. final dense layer output

    dw5 = dout * lorentzDx0(z2.T, w5, gamma) # loss gradient of final dense layer weights
    
    dz2 = lorentzDx(z2.T, w5, gamma).T.dot(dout) # loss gradient of first dense layer outputs 

    dw4 = dz2 * lorentzDx0(z1.T, w4, gamma)
    
    dz1 = lorentzDx(z1.T, w4, gamma).T.dot(dz2) # loss gradients of fully-connected layer

    dw3 = dz1 * lorentzDx0(fc.T, w3, gamma)

    dfc = lorentzDx(fc.T, w3, gamma).T.dot(dz1)
    dconv2 = dfc.reshape(conv2.shape) # reshape fully connected into dimensions of pooling layer
    
    dconv1, df2 = convolutionLorentzBackward(dconv2, conv1, f2, gamma, conv_s) # backpropagate previous gradient through second convolutional layer.
    
    dimage, df1 = convolutionLorentzBackward(dconv1, image, f1, gamma, conv_s) # backpropagate previous gradient through first convolutional layer.
    
    grads = [df1, df2, dw3, dw4, dw5] 
    
    return grads, loss, a1.flatten(), a2.flatten(), a3.flatten()

#####################################################
################### Optimization ####################
#####################################################

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost, config):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, w3, w4, w5] = params

    [num_filt1, num_filt2, conv_s, gamma] = config
    
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1] # get batch labels
    
    cost_ = 0
    batch_size = len(batch)
    
    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    dw5 = np.zeros(w5.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    v5 = np.zeros(w5.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    s5 = np.zeros(w5.shape)
    
    for i in range(batch_size):
        
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot
        
        # Collect Gradients for training example

        grads, loss, nl1, nl2, nl3 = conv(x, y, params, config)
        [df1_, df2_, dw3_, dw4_, dw5_] = grads
        
        df1 += df1_
        df2 += df2_

        dw3 += dw3_
        dw4 += dw4_
        dw5 += dw5_

        cost_+= loss

    # Parameter Update  
        
    v1 = beta1*v1 + (1-beta1)*df1/batch_size # momentum update
    s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMSProp update
    f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
   
    v2 = beta1*v2 + (1-beta1)*df2/batch_size
    s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * dw3/batch_size
    s3 = beta2*s3 + (1-beta2)*(dw3/batch_size)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4/batch_size
    s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)

    v5 = beta1*v5 + (1-beta1) * dw5/batch_size
    s5 = beta2*s5 + (1-beta2)*(dw5/batch_size)**2
    w5 -= lr * v5 / np.sqrt(s5+1e-7)
    

    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, w5]
    
    return params, cost, nl1, nl2, nl3


def gradDescent(batch, num_classes, lr, dim, n_c, params, cost, config):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, w3, w4, w5] = params

    [num_filt1, num_filt2, conv_s, gamma] = config
    
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1] # get batch labels
    
    cost_ = 0
    batch_size = len(batch)
    
    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    dw5 = np.zeros(w5.shape)
    
    for i in range(batch_size):
        
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot
        
        # Collect Gradients for training example

        grads, loss, nl1, nl2, nl3 = conv(x, y, params, config)
        [df1_, df2_, dw3_, dw4_, dw5_] = grads
        
        df1 += df1_
        df2 += df2_
        dw3 += dw3_

        dw4 += dw4_
        dw5 += dw5_

        cost_ += loss

    # Parameter Update  
        
    f1 -= lr * df1 / batch_size # simply gradient descent
   
    f2 -= lr * df2 / batch_size

    f3 -= lr * df3 / batch_size
    
    w4 -= lr * dw4 / batch_size
    
    w5 -= lr * dw5 / batch_size
    

    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, f3, w4, w5]
    
    return params, cost, nl1, nl2, nl3

#####################################################
##################### Training ######################
#####################################################

def train(num_classes = 10, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 8, img_depth = 1, f = 3, num_filt1 = 8, num_filt2 = 8, gamma = 2/np.pi, batch_size = 50, num_epochs = 1000, save_path = 'params.pkl', save = True):

    # training data
    X, y_dash = generateDataset()

    X -= np.mean(X)
    X /= np.std(X)
    train_data = np.hstack((X,y_dash))
    
    np.random.shuffle(train_data)

    num_filt3 = num_filt2
    pool_f = 2

    num_conv_layers = 2
    flattened_size = ((img_dim - num_conv_layers*(f - 1))**2) * num_filt2

    hidden_layer1 = 64
    hidden_layer2 = 64

    ## Initializing all the parameters
    f1, f2 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f)
    w3, w4, w5 = (hidden_layer1, flattened_size), (hidden_layer2, hidden_layer1), (num_classes, hidden_layer2)

    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    w3 = initializeFilter(w3)
    w4 = initializeWeight(w4)
    w5 = initializeWeight(w5)

    params = [f1, f2, w3, w4, w5]

    conv_s = 1

    config = [num_filt1, num_filt2, conv_s, gamma]

    cost = []

    nl1_m = []
    nl1_std = []
    nl2_m = []
    nl2_std = []
    nl3_m = []
    nl3_std = []

    print("LR: "+str(lr)+", Batch Size: "+str(batch_size)+", Gamma: "+str(gamma))

    t = tqdm(range(num_epochs))

    for epoch in enumerate(t):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        for batch in batches:
            params, cost, nl1, nl2, nl3 = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost, config)
            t.set_description("Cost: %.2f" % (cost[-1]))

            nl1_m.append(np.mean(nl1))
            nl1_std.append(np.std(nl1))

            nl2_m.append(np.mean(nl2))
            nl2_std.append(np.std(nl2))

            nl3_m.append(np.mean(nl3))
            nl3_std.append(np.std(nl3))

    layer_mean = [nl1_m, nl2_m, nl3_m]
    layer_std = [nl1_std, nl2_std, nl3_std]
    final_layer = [nl1, nl2, nl3]

    if save:    
        to_save = [params, cost, layer_mean, layer_std, final_layer]
        
        with open(save_path, 'wb') as file:
            pickle.dump(to_save, file)
        
    return cost
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

def conv(image, label, params, gamma):
    
    # [f1, f2, w3, w4, w5] = params 
    [f1, f2, w3, w4] = params
    # [num_filt1, num_filt2, conv_s, gamma, batch_size] = config
    
    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1, nl_store1 = convolutionLorentzBatch2(image, f1, gamma) # convolution operation
    
    conv2, nl_store2 = convolutionLorentzBatch2(conv1, f2, gamma) # second convolution operation
    
    (batch_size, nf2, dim2, _) = conv2.shape
    fc = conv2.reshape((batch_size, nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    a1 = lorentz(np.transpose(fc, axes=(0,2,1)), np.expand_dims(w3, axis=0), gamma)
    z1 = np.sum(a1, axis=2).reshape((batch_size, -1, 1))

    a2 = lorentz(np.transpose(z1, axes=(0,2,1)), np.expand_dims(w4, axis=0), gamma)
    out = np.sum(a2, axis=2).reshape((batch_size, -1, 1))
    # z2 = np.sum(a2, axis=2).reshape((batch_size, -1, 1))
    
    # a3 = lorentz(np.transpose(z2, axes=(0,2,1)), np.expand_dims(w5, axis=0), gamma)
    # out = np.sum(a3, axis=2).reshape((batch_size, -1, 1))

    probs = softmaxBatch(out) # predict class probabilities with the softmax activation function
    
    ################################################
    #################### Loss ######################
    ################################################
    
    loss = categoricalCrossEntropyBatch(probs, label) # categorical cross-entropy loss
        
    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label # derivative of loss w.r.t. final dense layer output

    # dl5 = lorentzDxWithBase(np.transpose(z2, axes=(0,2,1)), w5, gamma, a3)
    # dw5 = dout * -dl5 # loss gradient of final dense layer weights
    # dz2 = np.matmul(np.transpose(dl5, axes=(0,2,1)), dout) # loss gradient of first dense layer outputs 

    # dl4 = lorentzDxWithBase(np.transpose(z1, axes=(0,2,1)), w4, gamma, a2)
    # dw4 = dz2 * -dl4
    # dz1 = np.matmul(np.transpose(dl4, axes=(0,2,1)), dz2) # loss gradients of fully-connected layer

    dl4 = lorentzDxWithBase(np.transpose(z1, axes=(0,2,1)), w4, gamma, a2)
    dw4 = dout * -dl4
    dz1 = np.matmul(np.transpose(dl4, axes=(0,2,1)), dout) # loss gradients of fully-connected layer

    dl3 = lorentzDxWithBase(np.transpose(fc, axes=(0,2,1)), w3, gamma, a1)
    dw3 = dz1 * -dl3
    dfc = np.matmul(np.transpose(dl3, axes=(0,2,1)), dz1)

    dconv2 = dfc.reshape(conv2.shape) # reshape fully connected into dimensions of pooling layer
    
    dconv1, df2 = convolutionLorentzBackwardBatch2(dconv2, conv1, f2, nl_store2, gamma) # backpropagate previous gradient through second convolutional layer.
    
    dimage, df1 = convolutionLorentzBackwardBatch2(dconv1, image, f1, nl_store1, gamma) # backpropagate previous gradient through first convolutional layer.
    
    df1 = np.mean(df1, axis=0)
    df2 = np.mean(df2, axis=0)
    dw3 = np.mean(dw3, axis=0)
    dw4 = np.mean(dw4, axis=0)
    # dw5 = np.mean(dw5, axis=0)

    loss = np.mean(loss)

    # grads = [df1, df2, dw3, dw4, dw5] 
    grads = [df1, df2, dw3, dw4]
    
    # return grads, loss, nl_store1.flatten(), nl_store2.flatten(), a1.flatten(), a2.flatten(), a3.flatten()
    return grads, loss, nl_store1.flatten(), nl_store2.flatten(), a1.flatten(), a2.flatten()

#####################################################
################### Optimization ####################
#####################################################

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost, gamma):
    '''
    update the parameters through Adam gradient descnet.
    '''
    # [f1, f2, w3, w4, w5] = params
    [f1, f2, w3, w4] = params

    # [num_filt1, num_filt2, conv_s, gamma, batch_size] = config
    
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
    # dw5 = np.zeros(w5.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    # v5 = np.zeros(w5.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    # s5 = np.zeros(w5.shape)
        
    x = X
    y = np.eye(num_classes)[Y.astype(int)].reshape(batch_size, num_classes, 1) # convert label to one-hot
    
    # Collect Gradients for training example

    # grads, loss, nl1, nl2, nl3, nl4, nl5 = conv(x, y, params, config)
    # [df1, df2, dw3, dw4, dw5] = grads

    grads, loss, nl1, nl2, nl3, nl4 = conv(x, y, params, gamma)
    [df1, df2, dw3, dw4] = grads

    cost_ = loss

    # Parameter Update  
        
    v1 = beta1*v1 + (1-beta1)*df1 # momentum update
    s1 = beta2*s1 + (1-beta2)*(df1)**2 # RMSProp update
    f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
   
    v2 = beta1*v2 + (1-beta1)*df2
    s2 = beta2*s2 + (1-beta2)*(df2)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * dw3
    s3 = beta2*s3 + (1-beta2)*(dw3)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4
    s4 = beta2*s4 + (1-beta2)*(dw4)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)

    # v5 = beta1*v5 + (1-beta1) * dw5
    # s5 = beta2*s5 + (1-beta2)*(dw5)**2
    # w5 -= lr * v5 / np.sqrt(s5+1e-7)
    

    cost.append(cost_)

    # params = [f1, f2, w3, w4, w5]
    
    # return params, cost, nl1, nl2, nl3, nl4, nl5

    params = [f1, f2, w3, w4]

    return params, cost, nl1, nl2, nl3, nl4


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

    # hidden_layer1 = 64
    # hidden_layer2 = 64
    hidden_layer = 128

    ## Initializing all the parameters
    f1, f2 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f)
    # w3, w4, w5 = (hidden_layer1, flattened_size), (hidden_layer2, hidden_layer1), (num_classes, hidden_layer2)
    w3, w4 = (hidden_layer, flattened_size), (num_classes, hidden_layer)

    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    w3 = initializeFilter(w3)
    w4 = initializeWeight(w4)
    # w5 = initializeWeight(w5)

    # params = [f1, f2, w3, w4, w5]
    params = [f1, f2, w3, w4]

    conv_s = 1

    config = [num_filt1, num_filt2, conv_s, gamma, batch_size]

    cost = []

    # nl1_m = []
    # nl1_std = []
    # nl2_m = []
    # nl2_std = []
    # nl3_m = []
    # nl3_std = []

    nl1_q5 = []
    nl1_q25 = []
    nl1_q50 = []
    nl1_q75 = []
    nl1_q95 = []

    nl2_q5 = []
    nl2_q25 = []
    nl2_q50 = []
    nl2_q75 = []
    nl2_q95 = []

    nl3_q5 = []
    nl3_q25 = []
    nl3_q50 = []
    nl3_q75 = []
    nl3_q95 = []

    nl4_q5 = []
    nl4_q25 = []
    nl4_q50 = []
    nl4_q75 = []
    nl4_q95 = []

    # nl5_q5 = []
    # nl5_q25 = []
    # nl5_q50 = []
    # nl5_q75 = []
    # nl5_q95 = []

    print("LR: "+str(lr)+", Batch Size: "+str(batch_size)+", Gamma: "+str(gamma))

    t = tqdm(range(num_epochs))

    for epoch in enumerate(t):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        for batch in batches:
            # params, cost, nl1, nl2, nl3, nl4, nl5 = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost, config)
            params, cost, nl1, nl2, nl3, nl4 = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost, gamma)
            t.set_description("Cost: %.2f" % (cost[-1]))

            # nl1_m.append(np.mean(nl1))
            # nl1_std.append(np.std(nl1))

            # nl2_m.append(np.mean(nl2))
            # nl2_std.append(np.std(nl2))

            # nl3_m.append(np.mean(nl3))
            # nl3_std.append(np.std(nl3))

            nl1_q5.append(np.percentile(nl1, 5))
            nl1_q25.append(np.percentile(nl1, 25))
            nl1_q50.append(np.percentile(nl1, 50))
            nl1_q75.append(np.percentile(nl1, 75))
            nl1_q95.append(np.percentile(nl1, 95))

            nl2_q5.append(np.percentile(nl2, 5))
            nl2_q25.append(np.percentile(nl2, 25))
            nl2_q50.append(np.percentile(nl2, 50))
            nl2_q75.append(np.percentile(nl2, 75))
            nl2_q95.append(np.percentile(nl2, 95))

            nl3_q5.append(np.percentile(nl3, 5))
            nl3_q25.append(np.percentile(nl3, 25))
            nl3_q50.append(np.percentile(nl3, 50))
            nl3_q75.append(np.percentile(nl3, 75))
            nl3_q95.append(np.percentile(nl3, 95))

            nl4_q5.append(np.percentile(nl4, 5))
            nl4_q25.append(np.percentile(nl4, 25))
            nl4_q50.append(np.percentile(nl4, 50))
            nl4_q75.append(np.percentile(nl4, 75))
            nl4_q95.append(np.percentile(nl4, 95))

            # nl5_q5.append(np.percentile(nl5, 5))
            # nl5_q25.append(np.percentile(nl5, 25))
            # nl5_q50.append(np.percentile(nl5, 50))
            # nl5_q75.append(np.percentile(nl5, 75))
            # nl5_q95.append(np.percentile(nl5, 95))

    # layer_mean = [nl1_m, nl2_m, nl3_m]
    # layer_std = [nl1_std, nl2_std, nl3_std]
    # final_layer = [nl1, nl2, nl3, nl4, nl5]

    # q5 = [nl1_q5, nl2_q5, nl3_q5, nl4_q5, nl5_q5]
    # q25 = [nl1_q25, nl2_q25, nl3_q25, nl4_q25, nl5_q25]
    # q50 = [nl1_q50, nl2_q50, nl3_q50, nl4_q50, nl5_q50]
    # q75 = [nl1_q75, nl2_q75, nl3_q75, nl4_q75, nl5_q75]
    # q95 = [nl1_q95, nl2_q95, nl3_q95, nl4_q95, nl5_q95]

    final_layer = [nl1, nl2, nl3, nl4]

    q5 = [nl1_q5, nl2_q5, nl3_q5, nl4_q5]
    q25 = [nl1_q25, nl2_q25, nl3_q25, nl4_q25]
    q50 = [nl1_q50, nl2_q50, nl3_q50, nl4_q50]
    q75 = [nl1_q75, nl2_q75, nl3_q75, nl4_q75]
    q95 = [nl1_q95, nl2_q95, nl3_q95, nl4_q95]

    if save:    
        to_save = [params, cost, q5, q25, q50, q75, q95, final_layer]
        
        with open(save_path, 'wb') as file:
            pickle.dump(to_save, file)
        
    return cost
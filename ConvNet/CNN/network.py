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

import numpy as np
import pickle
from tqdm import tqdm
#####################################################
############### Building The Network ################
#####################################################

def conv(image, label, params, config):
    
    [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params 
    [num_filt1, num_filt2, num_filt3, conv_s, pool_f, pool_s, gamma] = config
    
    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolution(image, f1, s = conv_s) # convolution operation
    nonlin1 = lorentz(conv1, b1.reshape(num_filt1, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    conv2 = convolution(nonlin1, f2, s = conv_s) # second convolution operation
    nonlin2 = lorentz(conv2, b2.reshape(num_filt2, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    conv3 = convolution(nonlin2, f3, s = pool_s) # using convolution with higher stride as pooling layer
    pooled = lorentz(conv3, b3.reshape(num_filt3, 1, 1), gamma)
    
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    z = w4.dot(fc) # first dense layer
    a = lorentz(z, b4, gamma) # pass through Lorentzian non-linearity
    
    out = w5.dot(a) + b5 # second dense layer

    probs = softmax(out) # predict class probabilities with the softmax activation function
    
    ################################################
    #################### Loss ######################
    ################################################
    
    loss = categoricalCrossEntropy(probs, label) # categorical cross-entropy loss
        
    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label # derivative of loss w.r.t. final dense layer output

    dw5 = dout * a.T # loss gradient of final dense layer weights
    db5 = dout # loss gradient of final dense layer biases
    
    da = w5.T.dot(dout) # loss gradient of first dense layer outputs 

    dl4 = lorentzDx(z, b4, gamma)

    dw4 = da * dl4 * fc.T
    db4 = da * lorentzDx0(z, b4, gamma)
    
    dfc = w4.T.dot(da * dl4) # loss gradients of fully-connected layer (pooling layer)
    dpool = dfc.reshape(pooled.shape) # reshape fully connected into dimensions of pooling layer
    
    dconv3 = dpool * lorentzDx(conv3, b3.reshape(num_filt3, 1, 1), gamma) # backpropagate through lorentzian
    db3 = np.mean(dpool * lorentzDx0(conv3, b3.reshape(num_filt3, 1, 1), gamma), axis=(1,2)) # find grad for bias
    dnonlin2, df3 = convolutionBackward(dconv3, conv2, f3, pool_s) # backpropagate previous gradient through third convolutional pooling layer.

    dconv2 = dnonlin2 * lorentzDx(conv2, b2.reshape(num_filt2, 1, 1), gamma) # backpropagate through lorentzian
    db2 = np.mean(dnonlin2 * lorentzDx0(conv2, b2.reshape(num_filt2, 1, 1), gamma), axis=(1,2)) # find grad for bias
    dnonlin1, df2 = convolutionBackward(dconv2, conv1, f2, conv_s) # backpropagate previous gradient through second convolutional layer.
    
    dconv1 = dnonlin1 * lorentzDx(conv1, b1.reshape(num_filt1, 1, 1), gamma) # backpropagate through lorentzian
    db1 = np.mean(dnonlin1 * lorentzDx0(conv1, b1.reshape(num_filt1, 1, 1), gamma), axis=(1,2)) # find grad for bias
    dimage, df1 = convolutionBackward(dconv1, image, f1, conv_s) # backpropagate previous gradient through first convolutional layer.
    
    grads = [df1, df2, df3, dw4, dw5, db1, db2, db3, db4, db5] 
    
    return grads, loss

#####################################################
################### Optimization ####################
#####################################################

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost, config):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params

    [num_filt1, num_filt2, num_filt3, conv_s, pool_f, pool_s, gamma] = config
    
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1] # get batch labels
    
    cost_ = 0
    batch_size = len(batch)
    
    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    df3 = np.zeros(f3.shape)
    dw4 = np.zeros(w4.shape)
    dw5 = np.zeros(w5.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    db5 = np.zeros(b5.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(f3.shape)
    v4 = np.zeros(w4.shape)
    v5 = np.zeros(w5.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    bv5 = np.zeros(b5.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(f3.shape)
    s4 = np.zeros(w4.shape)
    s5 = np.zeros(w5.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
    bs5 = np.zeros(b5.shape)
    
    for i in range(batch_size):
        
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot
        
        # Collect Gradients for training example

        conv_s = 1
        pool_f = 2
        pool_s = 2
        gamma = 1

        grads, loss = conv(x, y, params, config)
        [df1_, df2_, df3_, dw4_, dw5_, db1_, db2_, db3_, db4_, db5_] = grads
        
        df1 += df1_
        db1 += db1_.reshape(b1.shape)
        df2 += df2_
        db2 += db2_.reshape(b2.shape)
        df3 += df3_
        db3 += db2_.reshape(b3.shape)

        dw4 += dw4_
        db4 += db4_
        dw5 += dw5_
        db5 += db5_

        cost_+= loss

    # Parameter Update  
        
    v1 = beta1*v1 + (1-beta1)*df1/batch_size # momentum update
    s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMSProp update
    f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
    
    bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
    bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
    b1 -= lr * bv1/np.sqrt(bs1+1e-7)
   
    v2 = beta1*v2 + (1-beta1)*df2/batch_size
    s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
                       
    bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
    bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
    b2 -= lr * bv2/np.sqrt(bs2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * df3/batch_size
    s3 = beta2*s3 + (1-beta2)*(df3/batch_size)**2
    f3 -= lr * v3/np.sqrt(s3+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
    bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4/batch_size
    s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
    bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)

    v5 = beta1*v5 + (1-beta1) * dw5/batch_size
    s5 = beta2*s5 + (1-beta2)*(dw5/batch_size)**2
    w5 -= lr * v5 / np.sqrt(s5+1e-7)
    
    bv5 = beta1*bv5 + (1-beta1)*db5/batch_size
    bs5 = beta2*bs5 + (1-beta2)*(db5/batch_size)**2
    b5 -= lr * bv5 / np.sqrt(bs5+1e-7)
    

    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5]
    
    return params, cost


def gradDescent(batch, num_classes, lr, dim, n_c, params, cost, config):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params

    [num_filt1, num_filt2, num_filt3, conv_s, pool_f, pool_s, gamma] = config
    
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1] # get batch labels
    
    cost_ = 0
    batch_size = len(batch)
    
    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    df3 = np.zeros(f3.shape)
    dw4 = np.zeros(w4.shape)
    dw5 = np.zeros(w5.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    db5 = np.zeros(b5.shape)
    
    for i in range(batch_size):
        
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot
        
        # Collect Gradients for training example

        conv_s = 1
        pool_f = 2
        pool_s = 2
        gamma = 1

        grads, loss = conv(x, y, params, config)
        [df1_, df2_, df3_, dw4_, dw5_, db1_, db2_, db3_, db4_, db5_] = grads
        
        df1 += df1_
        db1 += db1_.reshape(b1.shape)
        df2 += df2_
        db2 += db2_.reshape(b2.shape)
        df3 += df3_
        db3 += db2_.reshape(b3.shape)

        dw4 += dw4_
        db4 += db4_
        dw5 += dw5_
        db5 += db5_

        cost_ += loss

    # Parameter Update  
        
    f1 -= lr * df1 / batch_size # simply gradient descent
    b1 -= lr * db1 / batch_size
   
    f2 -= lr * df2 / batch_size
    b2 -= lr * db2 / batch_size

    f3 -= lr * df3 / batch_size
    b3 -= lr * db3 / batch_size
    
    w4 -= lr * dw4 / batch_size
    b4 -= lr * db4 / batch_size
    
    w5 -= lr * dw5 / batch_size
    b5 -= lr * db5 / batch_size
    

    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5]
    
    return params, cost

#####################################################
##################### Training ######################
#####################################################

def train(num_classes = 10, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 28, img_depth = 1, f = 5, num_filt1 = 8, num_filt2 = 8, batch_size = 32, num_epochs = 1, save_path = 'params.pkl'):

    # training data
    m = 50000
    X = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
    X-= int(np.mean(X))
    X/= int(np.std(X))
    train_data = np.hstack((X,y_dash))
    
    np.random.shuffle(train_data)

    num_filt3 = num_filt2
    pool_f = 2

    num_conv_layers = 2
    flat_size = (((img_dim - num_conv_layers*(f - 1))//(2*2//2))**2) * num_filt3

    full_layer_size = 128

    ## Initializing all the parameters
    f1, f2, f3 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f), (num_filt3, num_filt2, pool_f, pool_f)
    w4, w5 = (full_layer_size, flat_size), (num_classes, full_layer_size)

    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    f3 = initializeFilter(f3)
    w4 = initializeWeight(w4)
    w5 = initializeWeight(w5)

    b1, b2, b3 = (num_filt1, 1), (num_filt2, 1), (num_filt3, 1)
    b4, b5 = (full_layer_size, 1), (num_classes, 1)

    b1 = initializeBias(b1)
    b2 = initializeBias(b2)
    b3 = initializeBias(b3)
    b4 = initializeBias(b4)
    b5 = initializeBias(b5)

    params = [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5]

    conv_s = 1
    pool_f = 2
    pool_s = 2
    gamma = 1 / full_layer_size

    config = [num_filt1, num_filt2, num_filt3, conv_s, pool_f, pool_s, gamma]

    cost = []

    print("LR:"+str(lr)+", Batch Size:"+str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x,batch in enumerate(t):
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost, config)
            t.set_description("Cost: %.2f" % (cost[-1]))
            
    to_save = [params, cost]
    
    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)
        
    return cost
        
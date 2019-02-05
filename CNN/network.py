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

from forward import *
from backward import *
from utils import *

import numpy as np
import pickle
from tqdm import tqdm
#####################################################
############### Building The Network ################
#####################################################

def conv(image, label, params, conv_s, pool_f, pool_s, gamma):
    
    [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params 
    
    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolution(image, f1, s = conv_s) # convolution operation
    nonlin1 = lorentz(conv1, b1, gamma) # pass through Lorentzian non-linearity
    
    conv2 = convolution(nonlin1, f2, s = conv_s) # second convolution operation
    nonlin2 = lorentz(conv2, b2, gamma) # pass through Lorentzian non-linearity
    
    conv3 = convolution(nonlin2, f3, s = pool_s) # using convolution with higher stride as pooling layer
    pooled = lorentz(conv3, b3, gamma)
    
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    z = w4.dot(fc) # first dense layer
    a = lorentz(z, b4, gamma) # pass through Lorentzian non-linearity
    
    out = w5.dot(z) + b5 # second dense layer
    
    # not using softmax as exponential cannot be implemented in optics
    
    ################################################
    #################### Loss ######################
    ################################################
    
    dout = loss = meanSquaredError(out, label) # mean squared loss - cost function
        
    ################################################
    ############# Backward Operation ###############
    ################################################

    dw5 = dout * a # loss gradient of final dense layer weights
    db5 = dout # loss gradient of final dense layer biases
    
    da = w5.T.dot(dout) # loss gradient of first dense layer outputs 

    dl4 = lorentzDx(z, b4, gamma)

    dw4 = dz * dl4 * fc
    db4 = dz * lorentzDx0(z, b4, gamma)
    
    dfc = w4.T.dot(dz * dl4) # loss gradients of fully-connected layer (pooling layer)
    dpool = dfc.reshape(pooled.shape) # reshape fully connected into dimensions of pooling layer
    
    dconv3 = dpool * lorentzDx(conv3, b3, gamma) # backpropagate through lorentzian
    db3 = dpool * lorentzDx0(conv3, b3, gamma) # find grad for bias
    dnonlin2, df3 = convolutionBackward(dconv3, conv2, f3, pool_s) # backpropagate previous gradient through third convolutional pooling layer.

    dconv2 = dnonlin2 * lorentzDx(conv2, b2, gamma) # backpropagate through lorentzian
    db2 = dnonlin2 * lorentzDx0(conv2, b2, gamma) # find grad for bias
    dnonlin1, df2 = convolutionBackward(dconv2, conv1, f2, conv_s) # backpropagate previous gradient through second convolutional layer.
    
    dconv1 = dnonlin1 * lorentzDx(conv1, b1, gamma) # backpropagate through lorentzian
    db1 = dnonlin1 * lorentzDx0(conv1, b1, gamma) # find grad for bias
    dimage, df1 = convolutionBackward(dconv1, image, f1, conv_s) # backpropagate previous gradient through first convolutional layer.
    
    grads = [df1, df2, df3, dw4, dw5, db1, db2, db3, db4, db5] 
    
    return grads, loss

#####################################################
################### Optimization ####################
#####################################################

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
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
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
    
    for i in range(batch_size):
        
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot
        
        # Collect Gradients for training example
        grads, loss = conv(x, y, params, 1, 2, 2)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads
        
        df1+=df1_
        db1+=db1_
        df2+=df2_
        db2+=db2_
        dw3+=dw3_
        db3+=db3_
        dw4+=dw4_
        db4+=db4_

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
    
    v3 = beta1*v3 + (1-beta1) * dw3/batch_size
    s3 = beta2*s3 + (1-beta2)*(dw3/batch_size)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
    bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4/batch_size
    s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
    bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)
    

    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    
    return params, cost

#####################################################
##################### Training ######################
#####################################################

def train(num_classes = 10, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 28, img_depth = 1, f = 5, num_filt1 = 8, num_filt2 = 8, batch_size = 32, num_epochs = 2, save_path = 'params.pkl'):

    # training data
    m = 50000
    X = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
    X-= int(np.mean(X))
    X/= int(np.std(X))
    train_data = np.hstack((X,y_dash))
    
    np.random.shuffle(train_data)

    num_filt3 = 1
    pool_f = 2

    num_conv_layers = 2
    flat_size = ((img_dim - num_conv_layers*(f - 1))**2) * num_filt2

    full_layer_size

    ## Initializing all the parameters
    f1, f2, f3 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (num_filt3, num_filt2, pool_f, pool_f)
    w4, w5 = (full_layer_size, flat_size), (num_classes, full_layer_size)

    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    f3 = initializeFilter(f3)
    w4 = initializeWeight(w3)
    w5 = initializeWeight(w4)

    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((f3.shape[0],1))
    b4 = np.zeros((w4.shape[0],1))
    b5 = np.zeros((w5.shape[0],1))

    params = [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5]

    cost = []

    print("LR:"+str(lr)+", Batch Size:"+str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x,batch in enumerate(t):
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))
            
    to_save = [params, cost]
    
    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)
        
    return cost
        
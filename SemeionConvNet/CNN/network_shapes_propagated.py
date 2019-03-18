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
import json
from tqdm import tqdm
#####################################################
############### Building The Network ################
#####################################################

def conv(image, label, params, gamma, validation = False):
    
    # [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5] = params 
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolutionBatch(image, f1) # convolution operation
    nonlin1 = nonlin(conv1, b1.reshape(1, -1, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    conv2 = convolutionBatch(nonlin1, f2) # second convolution operation
    pooled = nonlin(conv2, b2.reshape(1, -1, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    (batch_size, nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((batch_size, nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    z1 = np.matmul(w3, fc)
    a1 = nonlin(z1, b3.reshape(1, -1, 1), gamma)

    # z2 = np.matmul(w4, a1) # first dense layer
    # a2 = lorentz(z2, b4.reshape(1,-1,1), gamma) # pass through Lorentzian non-linearity
    
    # out = np.matmul(w5, a2) + b5.reshape(1,-1,1) # second dense layer

    out = np.matmul(w4, a1) + b4.reshape(1,-1,1)

    probs = softmaxBatch(out) # predict class probabilities with the softmax activation function
    
    ################################################
    #################### Loss ######################
    ################################################
    
    loss = categoricalCrossEntropyBatch(probs, label) # categorical cross-entropy loss

    loss = np.mean(loss)

    if validation:
        return loss
        
    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label # derivative of loss w.r.t. final dense layer output

    # dw5 = dout * np.transpose(a2, (0,2,1)) # loss gradient of final dense layer weights
    # db5 = dout # loss gradient of final dense layer biases
    
    # da2 = np.matmul(w5.T, dout) # loss gradient of first dense layer outputs 

    # dl4 = lorentzDxWithBase(z2, b4.reshape(1,-1,1), gamma, a2)

    # dw4 = da2 * dl4 * np.transpose(a1, (0,2,1))
    # db4 = da2 * -dl4
    
    # da1 = np.matmul(w4.T, da2 * dl4) # loss gradients of fully-connected layer

    dw4 = dout * np.transpose(a1, (0,2,1)) 
    db4 = dout

    da1 = np.matmul(w4.T, dout)

    # dl3 = nonlinDxWithBase(z1, b3.reshape(1, -1, 1), gamma, a1)
    # dl03 = nonlinDx0WithBase(z1, b3.reshape(1, -1, 1), gamma, a1)
    dl3 = nonlinDx(z1, b3.reshape(1, -1, 1), gamma)
    dl03 = nonlinDx0(z1, b3.reshape(1, -1, 1), gamma)

    dal = da1 * dl3

    dw3 = dal * np.transpose(fc, (0,2,1))
    db3 = da1 * dl03 # lorentzian derivative with respect to x0 is minus that of with respect to x

    dfc = np.matmul(w3.T, dal)
    dpool = dfc.reshape(pooled.shape) # reshape fully connected into dimensions of pooling layer
    
    # dl2 = nonlinDxWithBase(conv2, b2.reshape(1, -1, 1, 1), gamma, pooled)
    # dl02 = nonlinDx0WithBase(conv2, b2.reshape(1, -1, 1, 1), gamma, pooled)
    dl2 = nonlinDx(conv2, b2.reshape(1, -1, 1, 1), gamma)
    dl02 = nonlinDx0(conv2, b2.reshape(1, -1, 1, 1), gamma)

    dconv2 = dpool * dl2 # backpropagate through lorentzian
    db2 = np.mean(dpool * dl02, axis=(2,3)) # find grad for bias
    dnonlin1, df2 = convolutionBackwardBatch(dconv2, conv1, f2) # backpropagate previous gradient through second convolutional layer.
    
    # dl1 = nonlinDxWithBase(conv1, b1.reshape(1, -1, 1, 1), gamma, nonlin1)
    # dl01 = nonlinDx0WithBase(conv1, b1.reshape(1, -1, 1, 1), gamma, nonlin1)
    dl1 = nonlinDx(conv1, b1.reshape(1, -1, 1, 1), gamma)
    dl01 = nonlinDx0(conv1, b1.reshape(1, -1, 1, 1), gamma)

    dconv1 = dnonlin1 * dl1 # backpropagate through lorentzian
    db1 = np.mean(dnonlin1 * dl01, axis=(2,3)) # find grad for bias
    df1 = convolutionBackwardBatch(dconv1, image, f1, final=True) # backpropagate previous gradient through first convolutional layer.
    
    df1 = np.mean(df1, axis=0)
    df2 = np.mean(df2, axis=0)
    dw3 = np.mean(dw3, axis=0)
    dw4 = np.mean(dw4, axis=0)
    # dw5 = np.mean(dw5, axis=0)
    db1 = np.mean(db1, axis=0).reshape(-1, 1)
    db2 = np.mean(db2, axis=0).reshape(-1, 1)
    db3 = np.mean(db3, axis=0)
    db4 = np.mean(db4, axis=0)
    # db5 = np.mean(db5, axis=0)

    

    # grads = [df1, df2, dw3, dw4, dw5, db1, db2, db3, db4, db5] 
    
    # return grads, loss, nonlin1, pooled, a1, a2

    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

    return grads, loss, nonlin1, pooled, a1

#####################################################
################### Optimization ####################
#####################################################

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost, gamma):
    '''
    update the parameters through Adam gradient descnet.
    '''
    # [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5] = params
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
    # dw5 = np.zeros(w5.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    # db5 = np.zeros(b5.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    # v5 = np.zeros(w5.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    # bv5 = np.zeros(b5.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    # s5 = np.zeros(w5.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
    # bs5 = np.zeros(b5.shape)
        
    x = X
    y = np.eye(num_classes)[Y.astype(int)].reshape(batch_size, num_classes, 1) # convert label to one-hot
    
    # Collect Gradients for training example

    # grads, loss, nl1, nl2, nl3, nl4 = conv(x, y, params, gamma)
    # [df1, df2, dw3, dw4, dw5, db1, db2, db3, db4, db5] = grads

    grads, loss, nl1, nl2, nl3 = conv(x, y, params, gamma)
    [df1, df2, dw3, dw4, db1, db2, db3, db4] = grads

    cost_ = loss

    # Parameter Update  
        
    v1 = beta1*v1 + (1-beta1)*df1 # momentum update
    s1 = beta2*s1 + (1-beta2)*(df1)**2 # RMSProp update
    f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
    
    bv1 = beta1*bv1 + (1-beta1)*db1
    bs1 = beta2*bs1 + (1-beta2)*(db1)**2
    b1 -= lr * bv1/np.sqrt(bs1+1e-7)
   
    v2 = beta1*v2 + (1-beta1)*df2
    s2 = beta2*s2 + (1-beta2)*(df2)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
                       
    bv2 = beta1*bv2 + (1-beta1) * db2
    bs2 = beta2*bs2 + (1-beta2)*(db2)**2
    b2 -= lr * bv2/np.sqrt(bs2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * dw3
    s3 = beta2*s3 + (1-beta2)*(dw3)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3
    bs3 = beta2*bs3 + (1-beta2)*(db3)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4
    s4 = beta2*s4 + (1-beta2)*(dw4)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4
    bs4 = beta2*bs4 + (1-beta2)*(db4)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)

    # v5 = beta1*v5 + (1-beta1) * dw5/batch_size
    # s5 = beta2*s5 + (1-beta2)*(dw5/batch_size)**2
    # w5 -= lr * v5 / np.sqrt(s5+1e-7)
    
    # bv5 = beta1*bv5 + (1-beta1)*db5/batch_size
    # bs5 = beta2*bs5 + (1-beta2)*(db5/batch_size)**2
    # b5 -= lr * bv5 / np.sqrt(bs5+1e-7)
    

    cost.append(cost_)

    # params = [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5]
    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    
    # return params, cost, nl1, nl2, nl3, nl4
    return params, cost, nl1, nl2, nl3


def gradDescent(batch, num_classes, lr, dim, n_c, params, cost, config):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5] = params

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
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    db5 = np.zeros(b5.shape)
    
    for i in range(batch_size):
        
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot
        
        # Collect Gradients for training example

        grads, loss, nl1, nl2, nl3, nl4 = conv(x, y, params, config)
        [df1_, df2_, dw3_, dw4_, dw5_, db1_, db2_, db3_, db4_, db5_] = grads
        
        df1 += df1_
        db1 += db1_.reshape(b1.shape)
        df2 += df2_
        db2 += db2_.reshape(b2.shape)
        dw3 += dw3_
        db3 += db3_.reshape(b3.shape)

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

def train(num_classes = 3, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 14, img_depth = 1, f = 5, num_filt1 = 8, num_filt2 = 8, gamma = 2/np.pi, batch_size = 64, num_epochs = 100,
          save_path = 'params', save = True, continue_training = False, old_save='params'):

    # training data
    X, y_dash = shapes_training_set()

    X -= np.mean(X)
    X /= np.std(X)
    train_data = np.hstack((X,y_dash))
    
    np.random.shuffle(train_data)

    X_val, y_dash_val = shapes_validation_set()

    X_val -= np.mean(X_val)
    X_val /= np.std(X_val)
    val_data = np.hstack((X_val,y_dash_val))
    
    np.random.shuffle(val_data)

    num_conv_layers = 2
    flattened_size = ((img_dim - num_conv_layers*(f - 1))**2) * num_filt2

    # hidden_layer1 = 64
    # hidden_layer2 = 64
    hidden_layer = 128

    if not continue_training:
        ## Initializing all the parameters
        f1, f2 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f)
        # w3, w4, w5 = (hidden_layer1, flattened_size), (hidden_layer2, hidden_layer1), (num_classes, hidden_layer2)
        w3, w4 = (hidden_layer, flattened_size), (num_classes, hidden_layer)

        f1 = initializeFilter(f1)
        f2 = initializeFilter(f2)
        w3 = initializeFilter(w3)
        w4 = initializeWeight(w4)
        # w5 = initializeWeight(w5)

        b1, b2 = (num_filt1, 1), (num_filt2, 1)
        # b3, b4, b5 = (hidden_layer1, 1), (hidden_layer2, 1), (num_classes, 1)
        b3, b4 = (hidden_layer, 1), (num_classes, 1)

        b1 = initializeBias(b1)
        b2 = initializeBias(b2)
        b3 = initializeBias(b3)
        b4 = initializeBias(b4)
        # b5 = initializeBias(b5)

        # params = [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5]
        params = [f1, f2, w3, w4, b1, b2, b3, b4]

    else:
        params, final_layer = pickle.load(open(old_save + '.pkl', 'rb'))

    cost = []
    cost_val = []
    
    conv_s = 1

    config = [num_filt1, num_filt2, conv_s, gamma, batch_size]

    

    # nl1_m = []
    # nl1_std = []
    # nl2_m = []
    # nl2_std = []
    # nl3_m = []
    # nl3_std = []
    # nl4_m = []
    # nl4_std = []

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

    # nl4_q5 = []
    # nl4_q25 = []
    # nl4_q50 = []
    # nl4_q75 = []
    # nl4_q95 = []

    print("LR: "+str(lr)+", Batch Size: "+str(batch_size)+", Gamma: "+str(gamma))

    t = tqdm(range(num_epochs))

    # checking for early stopping
    min_val = float('inf')
    PATIENCE = 500
    num_since_best = 0
    num_epochs = 0

    for epoch in t:

        # calculate loss on validation set
        np.random.shuffle(val_data)
        val_batch = val_data[:batch_size]

        X_val = val_batch[:,0:-1] # get batch inputs
        x_val = X_val.reshape(-1, img_depth, img_dim, img_dim)

        Y_val = val_batch[:,-1] # get batch labels
        y_val = np.eye(num_classes)[Y_val.astype(int)].reshape(-1, num_classes, 1) # convert label to one-hot

        c_val = conv(x_val, y_val, params, gamma, validation=True)

        cost_val.append(c_val)

        if c_val < min_val:
            min_val = c_val
            best_params = params
            num_since_best = 0
            num_epochs = epoch
        else:
            if num_since_best > PATIENCE:
                print()
                print("Early stopping due to non improvement of validation accuracy")
                break
            else:
                num_since_best += 1

        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        for batch in batches:
            # params, cost, nl1, nl2, nl3, nl4 = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost, gamma)
            params, cost, nl1, nl2, nl3 = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost, gamma)
            t.set_description("Training Cost: %.2f, Validation Cost: %.2f" % (cost[-1], cost_val[-1]))

            # nl1_m.append(np.mean(nl1))
            # nl1_std.append(np.std(nl1))

            # nl2_m.append(np.mean(nl2))
            # nl2_std.append(np.std(nl2))

            # nl3_m.append(np.mean(nl3))
            # nl3_std.append(np.std(nl3))

            # nl4_m.append(np.mean(nl4))
            # nl4_std.append(np.std(nl4))

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

            # nl4_q5.append(np.percentile(nl4, 5))
            # nl4_q25.append(np.percentile(nl4, 25))
            # nl4_q50.append(np.percentile(nl4, 50))
            # nl4_q75.append(np.percentile(nl4, 75))
            # nl4_q95.append(np.percentile(nl4, 95))

    final_layer = [nl1, nl2, nl3]

    layer_q5 = [nl1_q5, nl2_q5, nl3_q5]
    layer_q25 = [nl1_q25, nl2_q25, nl3_q25]
    layer_q50 = [nl1_q50, nl2_q50, nl3_q50]
    layer_q75 = [nl1_q75, nl2_q75, nl3_q75]
    layer_q95 = [nl1_q95, nl2_q95, nl3_q95]

    if save:
        to_save_pickle = [params, final_layer]
        to_save_json = [cost, cost_val, layer_q5, layer_q25, layer_q50, layer_q75, layer_q95]
        # to_save = [params, cost, cost_val]

        with open(save_path + '.pkl', 'wb') as file:
            pickle.dump(to_save_pickle, file)
        
        with open(save_path + '.json', 'w') as file:
            json.dump(to_save_json, file)
        
    return cost
        
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

from NN.forward import *
from NN.backward import *
from NN.utils import *

import numpy as np
import pickle
import copy
# from tqdm import tqdm
#####################################################
############### Building The Network ################
#####################################################

def conv(image, label, params, gamma, validation = False):
    
    # [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5] = params 
    [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params

    ################################################
    ############## Forward Operation ###############
    ################################################

    conv1 = convolutionComplexBatch(image, f1) # convolution operation
    nonlin1 = nonlinComplex(conv1, b1.reshape(1, -1, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    conv2 = convolutionComplexBatch(nonlin1, f2) # second convolution operation
    nonlin2 = nonlinComplex(conv2, b2.reshape(1, -1, 1, 1), gamma) # pass through Lorentzian non-linearity

    conv3 = convolutionComplexBatch(nonlin2, f3, s=2) # second convolution operation
    pooled = nonlinComplex(conv3, b3.reshape(1, -1, 1, 1), gamma) # pass through Lorentzian non-linearity
    
    (batch_size, nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((batch_size, nf2 * dim2 * dim2, 1)) # flatten pooled layer

    z1 = np.matmul(w4, fc) # second convolution operation
    a1 = nonlinComplex(z1, b4.reshape(1,-1,1), gamma) # pass through Lorentzian non-linearity

    # z2 = np.matmul(w4, a1) # first dense layer
    # a2 = lorentz(z2, b4.reshape(1,-1,1), gamma) # pass through Lorentzian non-linearity
    
    # out = np.matmul(w5, a2) + b5.reshape(1,-1,1) # second dense layer

    out = np.matmul(w5, a1) + b5.reshape(1,-1,1)

    measured = np.real(out)**2 + np.imag(out)**2

    probs = softmaxBatch(measured) # predict class probabilities with the softmax activation function
    
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
    dmeasured = probs - label # derivative of loss due to cross entropy and softmax

    # dw5 = dout * np.transpose(a2, (0,2,1)) # loss gradient of final dense layer weights
    # db5 = dout # loss gradient of final dense layer biases
    
    # da2 = np.matmul(w5.T, dout) # loss gradient of first dense layer outputs 

    # dl4 = lorentzDxWithBase(z2, b4.reshape(1,-1,1), gamma, a2)

    # dw4 = da2 * dl4 * np.transpose(a1, (0,2,1))
    # db4 = da2 * -dl4
    
    # da1 = np.matmul(w4.T, da2 * dl4) # loss gradients of fully-connected layer

    dout = 2 * out

    outmeasured = dout * dmeasured

    db5 = dout
    dw5 = outmeasured * np.transpose(a1, (0,2,1)).conj()

    da1 = np.matmul(w5.conj().T, outmeasured)

    dzr, dzi, dbr, dbi = nonlinComplexDxDx0Split(z1, b4.reshape(1,-1,1), gamma)

    az = da1.real * dzr + 1j * da1.imag * dzi

    db4 = da1.real * dbr + 1j * da1.imag * dbi
    dw4 = az * np.transpose(fc, (0,2,1)).conj()

    dfc = np.matmul(w4.conj().T, az)
    dpool = dfc.reshape(pooled.shape)

    dzr, dzi, dbr, dbi = nonlinComplexDxDx0Split(conv3, b3.reshape(1,-1,1,1), gamma)
    dconv3 = dpool.real * dzr + 1j * dpool.imag * dzi
    db3 = np.mean(dpool.real * dbr + 1j * dpool.imag * dbi, axis=(2,3))
    dnonlin2, df3 = convolutionComplexBackwardBatch(dconv3, conv2, f3, s=2) # backpropagate previous gradient through second convolutional layer.

    dzr, dzi, dbr, dbi = nonlinComplexDxDx0Split(conv2, b2.reshape(1,-1,1,1), gamma)
    dconv2 = dnonlin2.real * dzr + 1j * dnonlin2.imag * dzi
    db2 = np.mean(dnonlin2.real * dbr + 1j * dnonlin2.imag * dbi, axis=(2,3))
    dnonlin1, df2 = convolutionComplexBackwardBatch(dconv2, conv1, f2) # backpropagate previous gradient through second convolutional layer.
    
    dzr, dzi, dbr, dbi = nonlinComplexDxDx0Split(conv1, b1.reshape(1,-1,1,1), gamma)
    dconv1 = dnonlin1.real * dzr + 1j * dnonlin1.imag * dzi
    db1 = np.mean(dnonlin1.real * dbr + 1j * dnonlin1.imag * dbi, axis=(2,3))
    df1 = convolutionComplexBackwardBatch(dconv1, image, f1, final=True) # backpropagate previous gradient through second convolutional layer.

    df1 = np.mean(df1, axis=0)
    df2 = np.mean(df2, axis=0)
    df3 = np.mean(df3, axis=0)
    dw4 = np.mean(dw4, axis=0)
    dw5 = np.mean(dw5, axis=0)
    db1 = np.mean(db1, axis=0).reshape(-1,1)
    db2 = np.mean(db2, axis=0).reshape(-1,1)
    db3 = np.mean(db3, axis=0).reshape(-1,1)
    db4 = np.mean(db4, axis=0)
    db5 = np.mean(db5, axis=0)

    # grads = [df1, df2, dw3, dw4, dw5, db1, db2, db3, db4, db5] 
    
    # return grads, loss, nonlin1, pooled, a1, a2

    grads = [df1, df2, df3, dw4, dw5, db1, db2, db3, db4, db5]

    return grads, loss, nonlin1, nonlin2, pooled, a1

#####################################################
################### Optimization ####################
#####################################################

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost, gamma):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5] = params
    
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
        
    x = X
    y = np.eye(num_classes)[Y.astype(int)].reshape(batch_size, num_classes, 1) # convert label to one-hot
    
    # Collect Gradients for training example

    grads, loss, nl1, nl2, nl3, nl4 = conv(x, y, params, gamma)
    [df1, df2, df3, dw4, dw5, db1, db2, db3, db4, db5] = grads

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
    
    v3 = beta1*v3 + (1-beta1) * df3
    s3 = beta2*s3 + (1-beta2)*(df3)**2
    f3 -= lr * v3/np.sqrt(s3+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3
    bs3 = beta2*bs3 + (1-beta2)*(db3)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4
    s4 = beta2*s4 + (1-beta2)*(dw4)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4
    bs4 = beta2*bs4 + (1-beta2)*(db4)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)

    v5 = beta1*v5 + (1-beta1) * dw5/batch_size
    s5 = beta2*s5 + (1-beta2)*(dw5/batch_size)**2
    w5 -= lr * v5 / np.sqrt(s5+1e-7)
    
    bv5 = beta1*bv5 + (1-beta1)*db5/batch_size
    bs5 = beta2*bs5 + (1-beta2)*(db5/batch_size)**2
    b5 -= lr * bv5 / np.sqrt(bs5+1e-7)
    

    cost.append(cost_)

    params = [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5]
    
    return params, cost, nl1, nl2, nl3, nl4


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

def train(num_classes = 4, lr = 0.01, beta1 = 0.95, beta2 = 0.99,
          img_dim = 20, img_depth = 1, f = 5, p = 2, num_filt1 = 8, num_filt2 = 8, num_filt3 = 8, gamma = 2/np.pi, layer = 128, batch_size = 256, max_epochs = 5000,
          save_path = 'params.pkl', save = True, continue_training = False, progress_bar = True):

    # training data
    X, y_dash = fourshapes_training_set()
    X_val, y_dash_val = fourshapes_validation_set()

    train_data = norm_stack_shuffle(X,y_dash, by_column=False)

    val_data = norm_stack_shuffle(X_val,y_dash_val, by_column=False)

    num_conv_layers = 2
    flattened_size = (((img_dim - num_conv_layers*(f - 1))**2) * num_filt3) // p**2

    hidden_layer = layer

    if not continue_training:
        ## Initializing all the parameters
        f1, f2, f3 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f), (num_filt3, num_filt2, p, p)
        w4, w5 = (hidden_layer, flattened_size), (num_classes, hidden_layer)

        f1 = initializeFilter(f1)
        f2 = initializeFilter(f2)
        f3 = initializeFilter(f3)
        w4 = initializeWeight(w4)
        w5 = initializeWeight(w5)

        b1, b2, b3 = (num_filt1, 1), (num_filt2, 1), (num_filt3, 1)
        b4, b5 = (hidden_layer, 1), (num_classes, 1)

        b1 = initializeBias(b1)
        b2 = initializeBias(b2)
        b3 = initializeBias(b3)
        b4 = initializeBias(b4)
        b5 = initializeBias(b5)

        params = [f1, f2, f3, w4, w5, b1, b2, b3, b4, b5]

        cost = []
        cost_val = []

    else:
        params, cost, cost_val = pickle.load(open(save_path, 'rb'))

    nl1_l = []
    nl2_l = []
    nl3_l = []

    nl1_r5 = []
    nl1_r25 = []
    nl1_r50 = []
    nl1_r75 = []
    nl1_r95 = []

    nl2_r5 = []
    nl2_r25 = []
    nl2_r50 = []
    nl2_r75 = []
    nl2_r95 = []

    nl3_r5 = []
    nl3_r25 = []
    nl3_r50 = []
    nl3_r75 = []
    nl3_r95 = []

    nl4_r5 = []
    nl4_r25 = []
    nl4_r50 = []
    nl4_r75 = []
    nl4_r95 = []

    nl1_i5 = []
    nl1_i25 = []
    nl1_i50 = []
    nl1_i75 = []
    nl1_i95 = []

    nl2_i5 = []
    nl2_i25 = []
    nl2_i50 = []
    nl2_i75 = []
    nl2_i95 = []

    nl3_i5 = []
    nl3_i25 = []
    nl3_i50 = []
    nl3_i75 = []
    nl3_i95 = []

    nl4_i5 = []
    nl4_i25 = []
    nl4_i50 = []
    nl4_i75 = []
    nl4_i95 = []

    print("LR: "+str(lr)+", Batch Size: "+str(batch_size)+", Gamma: "+str(gamma))

    if progress_bar:
        t = tqdm(range(max_epochs))
    else:
        t = range(max_epochs)

    # checking for early stopping
    min_val = float('inf')
    PATIENCE = 10
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

        if c_val < min_val:
            min_val = c_val
            best_params = copy.deepcopy(params)
            num_since_best = 0
            num_epochs = epoch
        else:
            if num_since_best > PATIENCE:
                print("Early stopping due to non improvement of validation accuracy")
                break
            else:
                num_since_best += 1

        cost_val.append(c_val)

        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        for batch in batches:
            params, cost, nl1, nl2, nl3, nl4 = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost, gamma)

            if progress_bar:
                t.set_description("Training Cost: %.2f, Validation Cost: %.2f" % (cost[-1], cost_val[-1]))
            else:
                print("Training Cost: %.2f, Validation Cost: %.2f" % (cost[-1], cost_val[-1]))

            nl1_r5.append(np.percentile(nl1.real, 5))
            nl1_r25.append(np.percentile(nl1.real, 25))
            nl1_r50.append(np.percentile(nl1.real, 50))
            nl1_r75.append(np.percentile(nl1.real, 75))
            nl1_r95.append(np.percentile(nl1.real, 95))

            nl2_r5.append(np.percentile(nl2.real, 5))
            nl2_r25.append(np.percentile(nl2.real, 25))
            nl2_r50.append(np.percentile(nl2.real, 50))
            nl2_r75.append(np.percentile(nl2.real, 75))
            nl2_r95.append(np.percentile(nl2.real, 95))

            nl3_r5.append(np.percentile(nl3.real, 5))
            nl3_r25.append(np.percentile(nl3.real, 25))
            nl3_r50.append(np.percentile(nl3.real, 50))
            nl3_r75.append(np.percentile(nl3.real, 75))
            nl3_r95.append(np.percentile(nl3.real, 95))

            nl4_r5.append(np.percentile(nl4.real, 5))
            nl4_r25.append(np.percentile(nl4.real, 25))
            nl4_r50.append(np.percentile(nl4.real, 50))
            nl4_r75.append(np.percentile(nl4.real, 75))
            nl4_r95.append(np.percentile(nl4.real, 95))

            nl1_i5.append(np.percentile(nl1.imag, 5))
            nl1_i25.append(np.percentile(nl1.imag, 25))
            nl1_i50.append(np.percentile(nl1.imag, 50))
            nl1_i75.append(np.percentile(nl1.imag, 75))
            nl1_i95.append(np.percentile(nl1.imag, 95))

            nl2_i5.append(np.percentile(nl2.imag, 5))
            nl2_i25.append(np.percentile(nl2.imag, 25))
            nl2_i50.append(np.percentile(nl2.imag, 50))
            nl2_i75.append(np.percentile(nl2.imag, 75))
            nl2_i95.append(np.percentile(nl2.imag, 95))

            nl3_i5.append(np.percentile(nl3.imag, 5))
            nl3_i25.append(np.percentile(nl3.imag, 25))
            nl3_i50.append(np.percentile(nl3.imag, 50))
            nl3_i75.append(np.percentile(nl3.imag, 75))
            nl3_i95.append(np.percentile(nl3.imag, 95))

            nl4_i5.append(np.percentile(nl4.imag, 5))
            nl4_i25.append(np.percentile(nl4.imag, 25))
            nl4_i50.append(np.percentile(nl4.imag, 50))
            nl4_i75.append(np.percentile(nl4.imag, 75))
            nl4_i95.append(np.percentile(nl4.imag, 95))

    final_layer = [nl1, nl2, nl3, nl4]

    # layer_q5 = [nl1_q5, nl2_q5]
    # layer_q25 = [nl1_q25, nl2_q25]
    # layer_q50 = [nl1_q50, nl2_q50]
    # layer_q75 = [nl1_q75, nl2_q75]
    # layer_q95 = [nl1_q95, nl2_q95]

    nl1_p = [nl1_r5, nl1_r25, nl1_r50, nl1_r75, nl1_r95, nl1_i5, nl1_i25, nl1_i50, nl1_i75, nl1_i95]
    nl2_p = [nl2_r5, nl2_r25, nl2_r50, nl2_r75, nl2_r95, nl2_i5, nl2_i25, nl2_i50, nl2_i75, nl2_i95]
    nl3_p = [nl3_r5, nl3_r25, nl3_r50, nl3_r75, nl3_r95, nl3_i5, nl3_i25, nl3_i50, nl3_i75, nl3_i95]
    nl4_p = [nl4_r5, nl4_r25, nl4_r50, nl4_r75, nl4_r95, nl4_i5, nl4_i25, nl4_i50, nl4_i75, nl4_i95]

    if save:    
        # to_save = [params, cost, cost_val, nl1_l, nl2_l]
        to_save = [best_params, cost, cost_val, nl1_p, nl2_p, nl3_p, nl4_p, final_layer]
        
        with open(save_path, 'wb') as file:
            pickle.dump(to_save, file)
        
    return cost
        
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
from NN.utils import *

import numpy as np
import pickle
import copy
from tqdm import tqdm
#####################################################
############### Building The Network ################
#####################################################

def network(batch, label, params, gamma, validation=False):
    
    # [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5] = params 
    [w1, w2, w3] = params

    batch_size = len(batch)

    if type(gamma) is not list:
        gamma = [gamma, gamma]
    
    ################################################
    ############## Forward Operation ###############
    ################################################
    a1 = lorentz(np.transpose(batch, axes=(0,2,1)), np.expand_dims(w1, axis=0), gamma[0])
    z1 = np.sum(a1, axis=2).reshape((batch_size, -1, 1))

    a2 = lorentz(np.transpose(z1, axes=(0,2,1)), np.expand_dims(w2, axis=0), gamma[1])
    z2 = np.sum(a2, axis=2).reshape((batch_size, -1, 1))

    a3 = lorentz(np.transpose(z2, axes=(0,2,1)), np.expand_dims(w3, axis=0), 1)
    out = np.sum(a3, axis=2).reshape((batch_size, -1, 1))

    # z2 = np.matmul(w4, a1) # first dense layer
    # a2 = lorentz(z2, b4.reshape(1,-1,1), gamma) # pass through Lorentzian non-linearity
    
    # out = np.matmul(w5, a2) + b5.reshape(1,-1,1) # second dense layer

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

    dl3 = lorentzDxWithBase(np.transpose(z2, axes=(0,2,1)), w3, 1, a3)
    dw3 = dout * -dl3
    dz2 = np.matmul(np.transpose(dl3, axes=(0,2,1)), dout) # loss gradients of fully-connected layer

    dl2 = lorentzDxWithBase(np.transpose(z1, axes=(0,2,1)), w2, gamma[1], a2)
    dw2 = dz2 * -dl2
    dz1 = np.matmul(np.transpose(dl2, axes=(0,2,1)), dz2)

    dl1 = lorentzDxWithBase(np.transpose(batch, axes=(0,2,1)), w1, gamma[0], a1)
    dw1 = dz1 * -dl1
    
    dw1 = np.mean(dw1, axis=0)
    dw2 = np.mean(dw2, axis=0)
    dw3 = np.mean(dw3, axis=0)
    # dw5 = np.mean(dw5, axis=0)
    
    # return grads, loss, nonlin1, pooled, a1, a2

    grads = [dw1, dw2, dw3]

    return grads, loss, a1.flatten(), a2.flatten(), a3.flatten()

#####################################################
################### Optimization ####################
#####################################################

def adamGD(batch, num_classes, lr, dim, beta1, beta2, params, cost, gamma):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [w1, w2, w3] = params
    
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(-1, dim, 1)
    Y = batch[:,-1] # get batch labels
    
    cost_ = 0
    batch_size = len(batch)
    
    # initialize gradients and momentum,RMS params
    dw1 = np.zeros(w1.shape)
    dw2 = np.zeros(w2.shape)
    dw3 = np.zeros(w3.shape)
    # dw4 = np.zeros(w4.shape)
    # dw5 = np.zeros(w5.shape)
    
    v1 = np.zeros(w1.shape)
    v2 = np.zeros(w2.shape)
    v3 = np.zeros(w3.shape)
    # v4 = np.zeros(w4.shape)
    # v5 = np.zeros(w5.shape)
    
    s1 = np.zeros(w1.shape)
    s2 = np.zeros(w2.shape)
    s3 = np.zeros(w3.shape)
    # s4 = np.zeros(w4.shape)
    # s5 = np.zeros(w5.shape)
        
    x = X
    y = np.eye(num_classes)[Y.astype(int)].reshape(batch_size, num_classes, 1) # convert label to one-hot
    
    # Collect Gradients for training example

    grads, loss, nl1, nl2, nl3 = network(x, y, params, gamma)
    [dw1, dw2, dw3] = grads

    cost_ = loss

    # Parameter Update  
        
    v1 = beta1*v1 + (1-beta1)*dw1 # momentum update
    s1 = beta2*s1 + (1-beta2)*(dw1)**2 # RMSProp update
    w1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
   
    v2 = beta1*v2 + (1-beta1)*dw2
    s2 = beta2*s2 + (1-beta2)*(dw2)**2
    w2 -= lr * v2/np.sqrt(s2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * dw3
    s3 = beta2*s3 + (1-beta2)*(dw3)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    # v4 = beta1*v4 + (1-beta1) * dw4
    # s4 = beta2*s4 + (1-beta2)*(dw4)**2
    # w4 -= lr * v4 / np.sqrt(s4+1e-7)

    # v5 = beta1*v5 + (1-beta1) * dw5/batch_size
    # s5 = beta2*s5 + (1-beta2)*(dw5/batch_size)**2
    # w5 -= lr * v5 / np.sqrt(s5+1e-7)
    

    cost.append(cost_)

    params = [w1, w2, w3]
    
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

def train(num_classes = 3, lr = 0.01, beta1 = 0.95, beta2 = 0.99,
          data_dim = 4, gamma = 2/np.pi, layers = [32,32], batch_size = 64, num_epochs = 2000,
          save_path = 'params.pkl', save = True, continue_training = False, progress_bar = True):

    # training data
    X, y_dash = iris_training_set()
    X_val, y_dash_val = iris_validation_set()

    train_data = norm_stack_shuffle(X,y_dash)

    val_data = norm_stack_shuffle(X_val,y_dash_val)

    hidden_layer1 = layers[0]
    hidden_layer2 = layers[1]

    if not continue_training:
        ## Initializing all the parameters
        w1, w2, w3 = (hidden_layer1, data_dim), (hidden_layer2, hidden_layer1), (num_classes, hidden_layer2)

        w1 = initializeWeight(w1)
        w2 = initializeWeight(w2)
        w3 = initializeWeight(w3)
        # w4 = initializeWeight(w4)
        # w5 = initializeWeight(w5)

        params = [w1, w2, w3]

        cost = []
        cost_val = []

    else:
        params, cost, cost_val = pickle.load(open(save_path, 'rb'))


    # nl1_q5 = []
    # nl1_q25 = []
    # nl1_q50 = []
    # nl1_q75 = []
    # nl1_q95 = []

    # nl2_q5 = []
    # nl2_q25 = []
    # nl2_q50 = []
    # nl2_q75 = []
    # nl2_q95 = []

    # nl3_q5 = []
    # nl3_q25 = []
    # nl3_q50 = []
    # nl3_q75 = []
    # nl3_q95 = []

    # nl4_q5 = []
    # nl4_q25 = []
    # nl4_q50 = []
    # nl4_q75 = []
    # nl4_q95 = []

    print("LR: "+str(lr)+", Batch Size: "+str(batch_size)+", Gamma: "+str(gamma))

    if progress_bar:
        t = tqdm(range(num_epochs))
    else:
        t = range(num_epochs)

    # checking for early stopping
    min_val = float('inf')
    PATIENCE = 500
    num_since_best = 0
    num_epochs = 0

    for epoch in t:

        # calculate loss on validation set
        X_val = val_data[:,0:-1] # get batch inputs
        x_val = X_val.reshape(-1, data_dim, 1)

        Y_val = val_data[:,-1] # get batch labels
        y_val = np.eye(num_classes)[Y_val.astype(int)].reshape(-1, num_classes, 1) # convert label to one-hot

        c_val = network(x_val, y_val, params, gamma, validation=True)

        cost_val.append(c_val)

        if c_val < min_val:
            min_val = c_val
            best_params = copy.deepcopy(params)
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
            params, cost, nl1, nl2, nl3 = adamGD(batch, num_classes, lr, data_dim, beta1, beta2, params, cost, gamma)

            if progress_bar:
                t.set_description("Cost: %.2f" % (cost[-1]))

            # nl1_m.append(np.mean(nl1))
            # nl1_std.append(np.std(nl1))

            # nl2_m.append(np.mean(nl2))
            # nl2_std.append(np.std(nl2))

            # nl3_m.append(np.mean(nl3))
            # nl3_std.append(np.std(nl3))

            # nl4_m.append(np.mean(nl4))
            # nl4_std.append(np.std(nl4))

            # nl1_q5.append(np.percentile(nl1, 5))
            # nl1_q25.append(np.percentile(nl1, 25))
            # nl1_q50.append(np.percentile(nl1, 50))
            # nl1_q75.append(np.percentile(nl1, 75))
            # nl1_q95.append(np.percentile(nl1, 95))

            # nl2_q5.append(np.percentile(nl2, 5))
            # nl2_q25.append(np.percentile(nl2, 25))
            # nl2_q50.append(np.percentile(nl2, 50))
            # nl2_q75.append(np.percentile(nl2, 75))
            # nl2_q95.append(np.percentile(nl2, 95))

            # nl3_q5.append(np.percentile(nl3, 5))
            # nl3_q25.append(np.percentile(nl3, 25))
            # nl3_q50.append(np.percentile(nl3, 50))
            # nl3_q75.append(np.percentile(nl3, 75))
            # nl3_q95.append(np.percentile(nl3, 95))

    # final_layer = [nl1, nl2, nl3]

    # layer_q5 = [nl1_q5, nl2_q5, nl3_q5]
    # layer_q25 = [nl1_q25, nl2_q25, nl3_q25]
    # layer_q50 = [nl1_q50, nl2_q50, nl3_q50]
    # layer_q75 = [nl1_q75, nl2_q75, nl3_q75]
    # layer_q95 = [nl1_q95, nl2_q95, nl3_q95]

    if save:    
        # to_save = [params, cost, layer_q5, layer_q25, layer_q50, layer_q75, layer_q95, final_layer]
        to_save = [params, cost, cost_val, num_epochs]
        
        with open(save_path, 'wb') as file:
            pickle.dump(to_save, file)
        
    return cost
        
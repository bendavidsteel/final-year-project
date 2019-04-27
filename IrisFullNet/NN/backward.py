'''
Description: backpropagation operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018

Altered by: Ben Steel
Date: 05/02/19
'''

import numpy as np

from CNN.utils import *

#####################################################
############### Backward Operations #################
#####################################################
        
def convolutionBackward(dconv_prev, conv_in, filt, s=1):
    '''
    Backpropagation through a convolutional layer. 
    '''
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape) 
    dfilt = np.zeros(filt.shape)
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f] 
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    
    return dout, dfilt

def convolutionBackwardBatch(dconv_prev, conv_in, filt, s=1):
    '''
    Backpropagation through a convolutional layer. 
    '''
    (n_f, n_c, f, _) = filt.shape
    (batch_size, _, orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape) 
    dfilt = np.zeros((batch_size,) + filt.shape)
    for curr_b in range(batch_size):
        for curr_f in range(n_f):
            # loop through all filters
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    # loss gradient of filter (used to update the filter)
                    dfilt[curr_b, curr_f] += dconv_prev[curr_b, curr_f, out_y, out_x] * conv_in[curr_b, :, curr_y:curr_y+f, curr_x:curr_x+f]
                    # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                    dout[curr_b, :, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_b, curr_f, out_y, out_x] * filt[curr_f] 
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
    
    return dout, dfilt


def convolutionLorentzBackward(dconv_prev, conv_in, filt, gamma, s):
    '''
    Backpropagation through a convolutional layer. 
    '''
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape) 
    dfilt = np.zeros(filt.shape)
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * lorentzDx0(conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f], filt[curr_f], gamma)
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * lorentzDx(conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f], filt[curr_f], gamma) 
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    
    return dout, dfilt


def maxpoolBackward(dpool, orig, f, s):
    '''
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    '''
    (n_c, orig_dim, _) = orig.shape
    
    dout = np.zeros(orig.shape)
    
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]
                
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return dout

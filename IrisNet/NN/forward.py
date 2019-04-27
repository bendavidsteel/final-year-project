'''
Description: forward operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018

Altered by: Ben Steel
Date: 05/02/19
'''
import numpy as np


#####################################################
################ Forward Operations #################
#####################################################


def convolution(image, filt, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions
    
    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
    
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((n_f,out_dim,out_dim))
    
    # convolve the filter over every part of the image, adding the bias at each step. 
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return out

def convolutionBatch(image, filt, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    batch_size, n_c, in_dim, _ = image.shape # image dimensions
    
    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
    
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((batch_size, n_f,out_dim,out_dim))
    
    # convolve the filter over every part of the image, adding the bias at each step. 
    for curr_b in range(batch_size):
        for curr_f in range(n_f):
            curr_y = out_y = 0
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                while curr_x + f <= in_dim:
                    out[curr_b, curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[curr_b, :,curr_y:curr_y+f, curr_x:curr_x+f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
        
    return out

def convolutionLorentz(image, filt, gamma, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions
    
    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
    
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((n_f,out_dim,out_dim))
    
    # convolve the filter over every part of the image, adding the bias at each step. 
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(lorentz(image[:,curr_y:curr_y+f, curr_x:curr_x+f], filt[curr_f], gamma))
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return out

def maxpool(image, f=2, s=2):
    '''
    Downsample `image` using kernel size `f` and stride `s`
    '''
    n_c, h_prev, w_prev = image.shape
    
    h = int((h_prev - f)/s)+1
    w = int((w_prev - f)/s)+1
    
    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled

def subsample(image, f=2, s=2):
    '''
    Downsample `image` using kernel size `f` and stride `s`
    TODO maybe add trainable parameters later
    '''
    n_c, h_prev, w_prev = image.shape
    
    h = int((h_prev - f)/s)+1
    w = int((w_prev - f)/s)+1
    
    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        # slide pool window over each part of the image and assign the sum of values at each step to the output
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.sum(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def softmaxBatch(X):
    out = np.exp(X)
    sum = np.sum(out, axis=1).reshape(-1,1,1)
    return out/sum

def categoricalCrossEntropy(probs, label):
    return -1*np.sum(label * np.log(np.abs(probs)))

def categoricalCrossEntropyBatch(probs, label):
    return -1*np.sum(label * np.log(np.abs(probs)), axis=1)

def categoricalCrossEntropyBackwards(probs, label):
    return -1 * label / probs

def meanSquaredError(probs, label):
    return np.sum(np.square(probs - label))

def meanSquaredErrorBackwards(probs, label):
    return 2*(probs - label)

def lorentz(x, x0, gamma):
    # lorentz function
    return (0.5*gamma)/(np.pi * (np.square(x - x0) + np.square(0.5*gamma)))

def lorentzDx(x, x0, gamma):
    # derivative of lorentz function with respect to x
    return (-16 * (x - x0) * gamma) / (np.pi * np.square(4*np.square(x - x0) + np.square(gamma)))

def lorentzDx0(x, x0, gamma):
    # derivative of lorentz function with respect to x0
    return (16 * (x - x0) * gamma) / (np.pi * np.square(4*np.square(x - x0) + np.square(gamma)))

def lorentzDxWithBase(x, x0, gamma, lx):
    # derivative of lorentz function with respect to x
    return -4*(x - x0)*(np.pi/gamma)*np.square(lx)



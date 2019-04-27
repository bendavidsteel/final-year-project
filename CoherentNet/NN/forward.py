'''
Description: forward operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018

Altered by: Ben Steel
Date: 06/03/19
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
    s = np.sum(out, axis=1).reshape(-1,1,1)
    return out/s

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

def lorentzComplex(x, x0, gamma):
    # complex lorentz function
    return (0.5*gamma)/((x - x0)*-1j + 0.5*gamma)

def lorentzDx(x, x0, gamma):
    # derivative of lorentz function with respect to x
    return (-16 * (x - x0) * gamma) / (np.pi * np.square(4*np.square(x - x0) + np.square(gamma)))

def lorentzDxComplex(x, x0, gamma):
    # derivative of complex lorentz function with respect to x
    return (2 * gamma * 1j) / np.square(gamma - 2 * 1j * (x - x0))

def lorentzDx0(x, x0, gamma):
    # derivative of lorentz function with respect to x0
    return (16 * (x - x0) * gamma) / (np.pi * np.square(4*np.square(x - x0) + np.square(gamma)))

def lorentzDx0Complex(x, x0, gamma):
    # derivative of complex lorentz function with respect to x
    return (-2 * gamma * 1j) / np.square(gamma - 2 * 1j * (x - x0))

def lorentzDxWithBase(x, x0, gamma, lx):
    # derivative of lorentz function with respect to x
    return -4*(x - x0)*(np.pi/gamma)*np.square(lx)

def lorentzDxWithBaseComplex(x, x0, gamma, lx):
    # derivative of complex lorentz function with respect to x
    return 1j*(2/gamma)*np.square(lx)

def nonlin(x, x0, gamma):
    # lorentz function
    return (0.5*gamma*x)/(np.square(x - x0) + np.square(0.5*gamma))

def nonlinDx(x, x0, gamma):
    # derivative of lorentz function with respect to x
    return (2* gamma * (gamma**2 - 4 * (x**2 - x0**2))) / np.square(4*np.square(x - x0) + np.square(gamma))

def nonlinDx0(x, x0, gamma):
    # derivative of lorentz function with respect to x0
    return (x * (x - x0) * gamma) / np.square(4*np.square(x - x0) + np.square(gamma))

def nonlinDxWithBase(x, x0, gamma, lx):
    # derivative of lorentz function with respect to x
    return 0.5 * gamma * ((lx/x)**2 - (lx/(0.5*gamma))**2 + ((x0*lx/(0.5*gamma*x))**2))

def nonlinDx0WithBase(x, x0, gamma, lx):
    return ((lx/(0.5*gamma*x))**2)*gamma*x*(x - x0)

def nonlinComplex(x, x0, gamma):
    # complex lorentz function
    return (0.5*gamma*x)/((x.real**2 + x.imag**2 - x0.real**2 - x0.imag**2)*-1j + 0.5*gamma)

def nonlinComplexDxConj(x, x0, gamma):
    num1 = ((0.5*gamma)**2)*(1-1j)
    num2 = ((x0.real**2) + (x0.imag**2)) * (1 + 1j)
    num3 = (x**2) * (1j - 1)

    denom1 = 0.5 * 1j * gamma
    denom2 = x.real**2 + x.imag**2 - x0**2 - x0**2

    return -1*(num1 + 0.5*gamma*(num2 + num3)) / (denom1 + denom2) ** 2

def nonlinComplexDx0Conj(x, x0, gamma):
    num = gamma * 1j * x * np.conj(x0)

    denom1 = 0.5 * 1j * gamma
    denom2 = x.real**2 + x.imag**2 - x0**2 - x0**2

    return num / (denom1 + denom2) ** 2

def nonlinComplexDxDx0Split(x, x0, gamma):
    x2 = x.real**2
    y2 = x.imag**2
    a2 = x0.real**2
    b2 = x0.imag**2

    xyab = x2 + y2 - a2 - b2
    xyab2 = xyab**2

    half_k2 = (0.5 * gamma)**2

    xy = x.real * x.imag

    denom = (half_k2 + xyab2) ** 2

    re1 = gamma * (half_k2 + xyab2) * ((0.25*gamma - xy) + 1j*(0.5*a2 + 0.5*b2 - 0.5*x2 - 1.5*y2))
    seq1 = 4*gamma * xyab * (0.25*gamma*x.real - 0.5*x.imag*xyab)
    re2 = seq1 * (x.real + 1j*x.imag)

    im1 = gamma * (half_k2 + xyab2) * ((0.25*gamma - xy) - 1j*(1.5*x2 + 0.5*y2 - 0.5*a2 - 0.5*b2))
    seq2 = 4*gamma * xyab * (0.25*gamma*x.imag + 0.5*x.real*xyab)
    im2 = seq2 * (x.real - 1j*x.imag)

    reX = (re1 - re2) / denom
    imX = (im1 - im2) / denom

    re1 = x.imag * gamma * (half_k2 + xyab2)
    re2 = seq1

    im1 = seq2
    im2 = x.real * gamma * (half_k2 + xyab2)

    reX0 = (x0.real + 1j*x0.imag) * (re1 + re2) / denom
    imX0 = (x0.imag - 1j*x0.real) * (im1 - im2) / denom

    return reX, imX, reX0, imX0

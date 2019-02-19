import numpy as np
import time

def lorentz(x, x0, gamma):
    # lorentz function
    return (0.5*gamma)/(np.pi * (np.square((x - x0)) + np.square(0.5*gamma)))

def lorentzDx(x, x0, gamma):
    # derivative of lorentz function with respect to x  
    return -4*(x - x0)*(np.pi/gamma)*np.square(lorentz(x, x0, gamma))

def lorentzDx0(x, x0, gamma):
    # derivative of lorentz function with respect to x0
    return 4*(x - x0)*(np.pi/gamma)*np.square(lorentz(x, x0, gamma))

def lorentzDxV2(x, x0, gamma):
    return (-16 * (x - x0) * gamma) / (np.pi * np.square(4*np.square(x - x0) + np.square(gamma)))

x = np.random.rand(1000, 1)
y = np.random.rand(1, 1000)
g = 1

t0 = time.time()
a = lorentzDx(x, y, g)
t1 = time.time()

t2 = time.time()
b = lorentzDxV2(x, y, g)
t3 = time.time()

print("Time for version 1: " + str(t1 - t0))
print("Time for version 2: " + str(t3 - t2))
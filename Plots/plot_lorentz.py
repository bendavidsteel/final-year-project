'''Author : Ben Steel
Date : 26/03/19'''

from forward import *
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import FormatStrFormatter

fig, ((ax1, ax3), (ax5, ax7)) = plt.subplots(nrows=2, ncols=2, figsize=(8,5))

ax2 = ax1.twinx()

x = np.arange(-10, 10, 0.001)
x0 = 0
gamma = 1
ax1.plot(x, lorentz(x, x0, gamma), color='b', linestyle=':')

ax2.plot(x, nonlin(x, x0, gamma), color='r', linestyle='--')

ax1.set_xlabel("x")
ax1.set_ylabel("L(x)", color='b')
ax2.set_ylabel("xL(x)", color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')

ax4 = ax3.twinx()

y1 = lorentzDx(x, x0, gamma)
ax3.plot(x, y1, color='b', linestyle=':')

y2 = nonlinDx(x, x0, gamma)
ax4.plot(x, y2, color='r', linestyle='--')

ax3.set_xlabel("x")
ax3.set_ylabel("L(x)", color='b')
ax4.set_ylabel("xL(x)", color='r')
ax3.tick_params(axis='y', labelcolor='b')
ax4.tick_params(axis='y', labelcolor='r')

ax6 = ax5.twinx()

x = np.arange(0, 20, 0.001)
x0 = 10
gamma = 1

ax5.plot(x, lorentz(x, x0, gamma), color='b', linestyle=':')

ax6.plot(x, nonlin(x, x0, gamma), color='r', linestyle='--')

ax5.set_xlabel("x")
ax5.set_ylabel("L(x)", color='b')
ax6.set_ylabel("xL(x)", color='r')
ax5.tick_params(axis='y', labelcolor='b')
ax6.tick_params(axis='y', labelcolor='r')

ax8 = ax7.twinx()

y1 = lorentzDx(x, x0, gamma)
ax7.plot(x, y1, color='b', linestyle=':')

y2 = nonlinDx(x, x0, gamma)
ax8.plot(x, y2, color='r', linestyle='--')

ax7.set_xlabel("x")
ax7.set_ylabel("L(x)", color='b')
ax8.set_ylabel("xL(x)", color='r')
ax7.tick_params(axis='y', labelcolor='b')
ax8.tick_params(axis='y', labelcolor='r')

"""# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4,4))

x = np.linspace(-10, 10, 1000)
x0 = 0

gamma = 1
# ax1.plot(x, lorentz(x, x0, gamma))
ax1.plot(x, lorentzDx(x, x0, gamma))

gamma = 2
# ax1.plot(x, lorentz(x, x0, gamma))
ax1.plot(x, lorentzDx(x, x0, gamma))

gamma = 5
# ax1.plot(x, lorentz(x, x0, gamma))
ax1.plot(x, lorentzDx(x, x0, gamma))

gamma = 10
# ax1.plot(x, lorentz(x, x0, gamma))
ax1.plot(x, lorentzDx(x, x0, gamma))

# ax1.set_xlabel("x")
ax1.set_xlabel("x")

# ax1.set_ylabel("L(x)")
ax1.set_ylabel("L'(x)")
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax1.legend([r'$\kappa = 1$', r'$\kappa = 2$', r'$\kappa = 5$', r'$\kappa = 10$'])"""

"""fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

x = np.linspace(-5, 5, 1000)
x0 = 0
gamma = 1

ax1.plot(x, lorentz(x, x0, gamma))
ax2.plot(x, lorentzDx(x, x0, gamma))

y = 1/(1 + np.exp(-1*x))
ax1.plot(x, y)
ax2.plot(x, y*(1 - y))

ax1.plot(x, np.tanh(x))
ax2.plot(x, 1.0 - np.tanh(x)**2)

ax1.plot(x, (x > 0) * x)
ax2.plot(x, (x > 0) * 1)

ax1.set_xlabel("x", fontsize=16)
ax2.set_xlabel("x", fontsize=16)

ax1.set_ylabel("f(x)", fontsize=16)
ax2.set_ylabel(r'$\frac{df(x)}{dx}$', fontsize=16)

ax1.set_ylim(-1.2, 1.8)

ax1.legend([r'Lorentzian', r'Logistic', r'Tanh', r'ReLU'], loc=2)"""

"""fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

x = np.linspace(-10, 10, 1000)
x0 = 0
gamma = 1

y = lorentzComplex(x, x0, gamma)
dy = lorentzDxComplex(x, x0, gamma)

ax1.plot(x, np.abs(y))
ax1.plot(x, np.angle(y))

ax2.plot(x, np.abs(dy))
ax2.plot(x, np.angle(dy))

ax1.set_xlabel("x")
ax2.set_xlabel("x")

ax1.set_ylabel("L(x)")
ax2.set_ylabel("Derivative of L(x)")

ax1.legend(['Re(L(x))', 'Im(L(x))'])"""

"""fig, axes = plt.subplots(nrows=2, ncols=2)

gamma = 1

x = np.linspace(-10, 10, 1000)
x0 = 0

y = nonlinComplex(x, x0, gamma)

axes[0,0].plot(x, y.real)
axes[0,0].plot(x, y.imag)

dyr, dyi, _, _ = nonlinComplexDxDx0Split(x, x0, gamma)

axes[0,1].plot(x, dyr)
axes[0,1].plot(x, dyi)

x = np.linspace(9, 11, 1000)
x0 = 10

y = nonlinComplex(x, x0, gamma)

axes[1,0].plot(x, y.real)
axes[1,0].plot(x, y.imag)

dyr, dyi, _, _ = nonlinComplexDxDx0Split(x, x0, gamma)

axes[1,1].plot(x, dyr)
axes[1,1].plot(x, dyi)

axes[0,0].set_xlabel("x")
axes[0,1].set_xlabel("x")
axes[1,0].set_xlabel("x")
axes[1,1].set_xlabel("x")

axes[0,0].set_ylabel("xL(x)")
axes[0,1].set_ylabel("d(xL(x))/dx")
axes[1,0].set_ylabel("xL(x)")
axes[1,1].set_ylabel("d(xL(x))/dx")"""

"""x = 0
x0 = np.linspace(-10, 10, 1000)

y = nonlinComplex(x, x0, gamma)

axes[1,0].plot(x, y.real)
axes[1,0].plot(x, y.imag)

dyr, dyi, _, _ = nonlinComplexDxDx0Split(x, x0, gamma)

axes[1,1].plot(x, dy.real)
axes[1,1].plot(x, dy.imag)

axes[0,0].set_xlabel("x")
axes[0,1].set_xlabel("x")
axes[1,0].set_xlabel(r'$x_{0}$')
axes[1,1].set_xlabel(r'$x_{0}$')

axes[0,0].set_ylabel("L(x)")
ax2.set_ylabel("Derivative of L(x)")

axes[0,0].legend(['Re(L(x))', 'Im(L(x))'])"""

"""# np.abs(t) = np.abs(t_p)
# np.abs(p)**2 + np.abs(t)**2 = 1

def transmission(f):
    p = 1
    t = 1
    t_p = 1
    delta = 1
    T = (t_p * t * np.exp(1j * 2 * np.pi * delta / f)) / (1 - (p ** 2) * np.exp(1j * 4 * np.pi * delta / f))
    return T

f = np.linspace(-5, 10, 1000)
t = transmission(f)

plt.plot(f, t)"""

plt.tight_layout()
plt.show()
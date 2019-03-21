from NN.forward import *
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

"""fig, ((ax1, ax3), (ax5, ax7)) = plt.subplots(nrows=2, ncols=2)

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

x = np.arange(1590, 1610, 0.001)
x0 = 1600
gamma = 1
ax5.plot(x, lorentz(x, x0, gamma), color='b', linestyle=':')

ax6.plot(x, nonlin(x, x0, gamma), color='r', linestyle='--')

ax5.set_xlabel("x")
ax5.set_ylabel("L(x)", color='b')
ax6.set_ylabel("xL(x)", color='r')
ax5.tick_params(axis='y', labelcolor='b')
ax6.tick_params(axis='y', labelcolor='r')

ax8 = ax7.twinx()

x = np.arange(1590, 1610, 0.001)
y1 = lorentzDx(x, x0, gamma)
ax7.plot(x, y1, color='b', linestyle=':')

y2 = nonlinDx(x, x0, gamma)
ax8.plot(x, y2, color='r', linestyle='--')

ax7.set_xlabel("x")
ax7.set_ylabel("L(x)", color='b')
ax8.set_ylabel("xL(x)", color='r')
ax7.tick_params(axis='y', labelcolor='b')
ax8.tick_params(axis='y', labelcolor='r')"""

"""fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

x = np.linspace(-10, 10, 1000)
x0 = 0

gamma = 1
ax1.plot(x, lorentz(x, x0, gamma))
ax2.plot(x, lorentzDx(x, x0, gamma))

gamma = 2
ax1.plot(x, lorentz(x, x0, gamma))
ax2.plot(x, lorentzDx(x, x0, gamma))

gamma = 5
ax1.plot(x, lorentz(x, x0, gamma))
ax2.plot(x, lorentzDx(x, x0, gamma))

gamma = 10
ax1.plot(x, lorentz(x, x0, gamma))
ax2.plot(x, lorentzDx(x, x0, gamma))

ax1.set_xlabel("x")
ax2.set_xlabel("x")

ax1.set_ylabel("L(x)")
ax2.set_ylabel("Derivative of L(x)")

ax2.legend([r'$\kappa = 1$', r'$\kappa = 2$', r'$\kappa = 5$', r'$\kappa = 10$'])"""

"""fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

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

ax1.set_xlabel("x")
ax2.set_xlabel("x")

ax1.set_ylabel("L(x)")
ax2.set_ylabel("Derivative of L(x)")

ax1.set_ylim(-1.2, 1.8)

ax1.legend([r'Lorentzian', r'Logistic', r'Tanh', r'ReLU'])"""

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

fig, axes = plt.subplots(nrows=2, ncols=2)

gamma = 1

x = np.linspace(-10, 10, 1000)
x0 = 0

y = nonlinComplex(x, x0, gamma)

axes[0,0].plot(x, y.real)
axes[0,0].plot(x, y.imag)

dy = nonlinDxComplex(x, x0, gamma)

axes[0,1].plot(x, dy.real)
axes[0,1].plot(x, dy.imag)

x = 0
x0 = np.linspace(-10, 10, 1000)

y = nonlinComplex(x, x0, gamma)

axes[1,0].plot(x, y.real)
axes[1,0].plot(x, y.imag)

dy = nonlinDx0Complex(x, x0, gamma)

axes[1,1].plot(x, dy.real)
axes[1,1].plot(x, dy.imag)

axes[0,0].set_xlabel("x")
axes[0,1].set_xlabel("x")
axes[1,0].set_xlabel(r'$x_{0}$')
axes[1,1].set_xlabel(r'$x_{0}$')

axes[0,0].set_ylabel("L(x)")
ax2.set_ylabel("Derivative of L(x)")

ax1.legend(['Re(L(x))', 'Im(L(x))'])

plt.tight_layout()
plt.show()
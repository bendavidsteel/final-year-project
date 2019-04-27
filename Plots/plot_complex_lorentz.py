from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from forward import *


fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = nonlinComplex(X + 1j*Y, 0, 1)

# Plot the surface.
surf = ax1.plot_wireframe(X, Y, np.real(Z), rstride=4, cstride=4)

surf = ax2.plot_wireframe(X, Y, np.imag(Z), rstride=4, cstride=4)

# Customize the z axis.
ax1.set_xlabel(r'$\Re(x)$')
ax1.set_ylabel(r'$\Im(x)$')
ax1.set_zlabel(r'$\Re(f(x))$')
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax2.set_xlabel(r'$\Re(x)$')
ax2.set_ylabel(r'$\Im(x)$')
ax2.set_zlabel(r'$\Im(f(x))$')
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
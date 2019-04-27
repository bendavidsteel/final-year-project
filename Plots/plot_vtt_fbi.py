'''Author : Ben Steel
Date : 19/03/19'''

import numpy as np 
import matplotlib.pyplot as plt

x_0v = []
y_0v = []

x_14v = []
y_14v = []

with open('bluedataset.csv', 'r') as f:
    for line in f:
            data = line.split(',')
            x_0v.append(float(data[0]))
            y_0v.append(float(data[1]))

with open('reddataset.csv', 'r') as f:
    for line in f:
            data = line.split(',')
            x_14v.append(float(data[0]))
            y_14v.append(float(data[1]))

plt.plot(x_0v, y_0v)
plt.plot(x_14v, y_14v)

plt.arrow(1557, 43, 30, 0, head_width=2)
plt.arrow(1643, 43, -30, 0, head_width=2)
plt.text(1650, 42, r'$\kappa$')
plt.plot([1590, 1590], [0, 100], color='k', linewidth=0.4)
plt.plot([1609, 1609], [0, 100], color='k', linewidth=0.4)
plt.plot([1600, 1600], [0, 100], color='k', linewidth=0.4)

plt.xlim([1200, 2100])
plt.ylim([0, 100])

plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission %")
plt.legend(["0V", "14V"])
plt.show()
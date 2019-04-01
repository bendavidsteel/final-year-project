'''Adapted from:
https://github.com/fancompute/neuroptica-notebooks/blob/master/neuroptica_demo.ipynb
By Ben Steel
Date: 25/03/19'''

import neuroptica as neu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def normalize_inputs(data, num_inputs, P0=10):
    '''
    Reshapes the inputs to fit into the specified mesh size and normalizes input data to 
    have the same total power input by injecting extra power to an "unused" input port.
    :param X: the input data
    :param num_inputs: the size of the network (number of waveguides)
    :param P0: the total power to inject with each data input
    '''
    _, input_size = data.shape
    injection_port = input_size
    data_normalized = np.array(np.pad(data, ((0, 0), (0, num_inputs - input_size)), mode="constant"))
    for i, x in enumerate(data_normalized):
        data_normalized[i][injection_port] = np.sqrt(P0 - np.sum(x**2))
    return data_normalized


def plot_planar_boundary(X, Y, model, grid_points=20, P0=10):
    '''
    Plots the decision boundary for a model predicting planar datasets
    :param X: shape (n_features, n_samples), first two features are x, y coordinates
    :param Y: true labeles, (n_features, n_samples)
    :param model: a trained neuroptica model
    :param grid_points: number of grid points to render
    :param P0: normalization power to shine into other ports to keep total power equal
    :return:
    '''
    
    labels = np.array([0 if yi[0] > yi[1] else 1 for yi in np.abs(Y)]).flatten() # True labels

    # Prepare a grid of inputs to predict
    x_min, y_min = np.min(X, axis=0)[0:2]
    x_max, y_max = np.max(X, axis=0)[0:2]
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points), np.linspace(x_min, x_max, grid_points))
    N = model.layers[0].input_size
    inputs = np.array([xx.flatten(), yy.flatten()]).T
    inputs = normalize_inputs(inputs, N, P0=P0).T

    # Predict the function value for the whole grid
    Y_hat = model.forward_pass(inputs).T
    Y_hat = [(0 if yhat[0] > yhat[1] else 1) for yhat in Y_hat]
    Z = np.array(Y_hat).reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure(figsize=(7.5, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap=plt.cm.Spectral)

    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    # # Convert one-hot Y labels to 0/1 decisions
    # labels = np.array([0 if yi[0] > yi[1] else 1 for yi in Y]).flatten()

    # plt.figure(figsize=(6,6))
    # plt.scatter((X.T)[0, :], (X.T)[1, :], c=labels, cmap=plt.cm.Spectral)
    # plt.colorbar()
    # plt.show()

    # fbi_activation = neu.FabryPerotInferometer(1)

    # x = np.linspace(-3, 3, 100).reshape((-1,1))
    # y = fbi_activation.forward_pass(x)
    # plt.plot(x, np.real(y),label="Re")
    # plt.plot(x, np.imag(y),label="Im")
    # plt.plot(x, np.abs(y), label="Abs")
    # plt.xlabel("Input field (a.u.)")
    # plt.ylabel("Output field (a.u.)")
    # plt.legend()
    # plt.show()

    X, Y = neu.utils.generate_ring_planar_dataset()
    labels = np.array([0 if yi[0] > yi[1] else 1 for yi in np.abs(Y)]).flatten() # True labels

    # Prepare a grid of inputs to predict
    grid_points = 20
    P0 = 10.0
    N = 5

    x_min, y_min = np.min(X, axis=0)[0:2]
    x_max, y_max = np.max(X, axis=0)[0:2]
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points), np.linspace(x_min, x_max, grid_points))
    inputs = np.array([xx.flatten(), yy.flatten()]).T
    inputs = normalize_inputs(inputs, N, P0=P0).T

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.set_ylabel('x2')
    ax1.set_xlabel('x1')
    ax1.set_xlim(left=x_min, right=x_max)
    ax1.set_ylim(bottom=y_min, top=y_max)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("$\mathcal{L}$")
    ax2.set_xlim(left=0, right=1000)
    ax2.set_ylim(bottom=0, top=1)

    model = neu.Sequential([
        neu.ClementsLayer(N),
        neu.Activation(neu.FabryPerotInferometer(N)),
        neu.ClementsLayer(N),
        neu.Activation(neu.FabryPerotInferometer(N)),
        neu.ClementsLayer(N),
        neu.Activation(neu.FabryPerotInferometer(N)),
        neu.ClementsLayer(N),
        neu.Activation(neu.FabryPerotInferometer(N)),
        neu.ClementsLayer(N),
        neu.Activation(neu.AbsSquared(N)), # photodetector measurement
        neu.DropMask(N, keep_ports=[0,1])
    ])

    X_normalized = normalize_inputs(X, N, P0=P0)

    optimizer = neu.InSituAdam(model, neu.CategoricalCrossEntropy, step_size=0.005)

    losses = []

    for epoch in range(1000):
        loss = optimizer.fit(X_normalized.T, Y.T, epochs=1, batch_size=32, show_progress=False)
        losses.append(loss)

        ax2.plot(losses)
        
        # Predict the function value for the whole grid
        Y_hat = model.forward_pass(inputs).T
        Y_hat = [(0 if yhat[0] > yhat[1] else 1) for yhat in Y_hat]
        Z = np.array(Y_hat).reshape(xx.shape)

        # Plot the contour and training examples
        im = ax1.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        ax1.scatter(X[:,0], X[:,1], c=labels, cmap=plt.cm.Spectral)
        # fig.colorbar(im, ax=ax1)

        plt.pause(0.05)

    
    plt.show()
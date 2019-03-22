''' Author: Ben Steel '''


from NN.network import *
from NN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle


if __name__ == '__main__':

    iters = 20

    gammas = np.linspace(0.1, 5, 5)

    for gamma in gammas:

        c = []

        max_len = 0
        min_len = np.inf

        for i in range(iters):
            cost = train(save = False, gamma = gamma, progress_bar=False)
            c.append(cost)

            if len(cost) > max_len:
                max_len = len(cost)

            if len(cost) < min_len:
                min_len = len(cost)

        costs = np.empty((len(c), max_len)) * np.nan

        for i in range(len(c)):
            for j in range(len(c[i])):
                costs[i][j] = c[i][j]

        q25_cost = np.nanpercentile(costs, 25, axis=0)
        q50_cost = np.nanpercentile(costs, 50, axis=0)
        q75_cost = np.nanpercentile(costs, 75, axis=0)

        to_save = [q25_cost, q50_cost, q75_cost, min_len, max_len]

        save_path = "gamma_error_curves_" + str(gamma) + "_heartdisease_coherentnet1616"

        with open(save_path, 'wb') as file:
            pickle.dump(to_save, file)

    
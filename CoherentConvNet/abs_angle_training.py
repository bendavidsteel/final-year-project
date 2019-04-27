'''Author: Ben Steel
Date: 14/03/19'''

from NN.network import *
from NN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_2overpiGamma_CoherentNet3224_nl_heartDataset'
    gamma = 1

    cost = train(gamma = gamma, save_path = save_path, continue_training = False)

    params, cost, cost_val, nl1, nl2 = pickle.load(open(save_path, 'rb'))

    [nl1_r5, nl1_r25, nl1_r50, nl1_r75, nl1_r95, nl1_t5, nl1_t25, nl1_t50, nl1_t75, nl1_t95] = nl1
    [nl2_r5, nl2_r25, nl2_r50, nl2_r75, nl2_r95, nl2_t5, nl2_t25, nl2_t50, nl2_t75, nl2_t95] = nl2
    
    legend = ["Layer 1", "Layer 2"]

    sub = 300
    i1 = 100
    n = len(nl1_r50)
    
    rmed1 = [nl1_r50[i] for i in range(0, n, sub)]
    rmed2 = [nl2_r50[i] for i in range(i1, n, sub)]

    tmed1 = [nl1_t50[i] for i in range(0, n, sub)]
    tmed2 = [nl2_t50[i] for i in range(i1, n, sub)]

    rq1 = [[(nl1_r50[i] - nl1_r25[i]) for i in range(0, n, sub)], [(nl1_r75[i] - nl1_r50[i]) for i in range(0, n, sub)]]
    rq2 = [[(nl2_r50[i] - nl2_r25[i]) for i in range(i1, n, sub)], [(nl2_r75[i] - nl2_r50[i]) for i in range(i1, n, sub)]]

    tq1 = [[(nl1_t50[i] - nl1_t25[i]) for i in range(0, n, sub)], [(nl1_t75[i] - nl1_t50[i]) for i in range(0, n, sub)]]
    tq2 = [[(nl2_t50[i] - nl2_t25[i]) for i in range(i1, n, sub)], [(nl2_t75[i] - nl2_t50[i]) for i in range(i1, n, sub)]]

    routlower1 = [nl1_r5[i] for i in range(0, n, sub)]
    routlower2 = [nl2_r5[i] for i in range(i1, n, sub)]

    toutlower1 = [nl1_t5[i] for i in range(0, n, sub)]
    toutlower2 = [nl2_t5[i] for i in range(i1, n, sub)]

    routupper1 = [nl1_r95[i] for i in range(0, n, sub)]
    routupper2 = [nl2_r95[i] for i in range(i1, n, sub)]

    toutupper1 = [nl1_t95[i] for i in range(0, n, sub)]
    toutupper2 = [nl2_t95[i] for i in range(i1, n, sub)]

    batches1 = [x for x in range(0, n, sub)]
    batches2 = [x for x in range(i1, n, sub)]

    plt.subplot(311)

    plt.plot(cost)
    plt.plot(np.linspace(0, len(cost), len(cost_val)), cost_val)
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')

    plt.subplot(312)

    plt.errorbar(batches1, rmed1, yerr=rq1, color='b')
    plt.errorbar(batches2, rmed2, yerr=rq2, color='g')

    plt.scatter(batches1, routlower1, color='b')
    plt.scatter(batches2, routlower2, color='g')

    plt.scatter(batches1, routupper1, color='b')
    plt.scatter(batches2, routupper2, color='g')

    plt.xlabel('Batch Updates')
    plt.ylabel('abs(Activation Value)')
    plt.legend(legend, loc='upper right')

    plt.subplot(313)

    plt.errorbar(batches1, tmed1, yerr=tq1, color='b')
    plt.errorbar(batches2, tmed2, yerr=tq2, color='g')

    plt.scatter(batches1, toutlower1, color='b')
    plt.scatter(batches2, toutlower2, color='g')

    plt.scatter(batches1, toutupper1, color='b')
    plt.scatter(batches2, toutupper2, color='g')

    plt.xlabel('Batch Updates')
    plt.ylabel('angle(Activation Value)')
    plt.legend(legend, loc='upper right')

    plt.show()
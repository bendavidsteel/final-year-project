'''Author: Ben Steel
Date: 16/03/19'''

import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_4Gamma_bias1b_CoherentNet3232_heartDataset_NLdata_try0'

    params, cost, cost_val, nl1_p, nl2_p, final_layer = pickle.load(open(save_path, 'rb'))

    [nl1_r5, nl1_r25, nl1_r50, nl1_r75, nl1_r95, nl1_i5, nl1_i25, nl1_i50, nl1_i75, nl1_i95] = nl1_p
    [nl2_r5, nl2_r25, nl2_r50, nl2_r75, nl2_r95, nl2_i5, nl2_i25, nl2_i50, nl2_i75, nl2_i95] = nl2_p

    # legend = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]
    legend = ["Layer 1", "Layer 2"]

    n = len(nl1_r50)
    sub = n // 40
    offset = sub // 2
    
    r_med1 = [nl1_r50[i] for i in range(0, n, sub)]
    r_med2 = [nl2_r50[i] for i in range(offset*1, n, sub)]
    # med3 = [nl3_q50[i] for i in range(offset*2, n, sub)]
    # med4 = [nl4_q50[i] for i in range(15, n, sub)]

    i_med1 = [nl1_i50[i] for i in range(0, n, sub)]
    i_med2 = [nl2_i50[i] for i in range(offset*1, n, sub)]

    r_q1 = [[(nl1_r50[i] - nl1_r25[i]) for i in range(0, n, sub)], [(nl1_r75[i] - nl1_r50[i]) for i in range(0, n, sub)]]
    r_q2 = [[(nl2_r50[i] - nl2_r25[i]) for i in range(offset*1, n, sub)], [(nl2_r75[i] - nl2_r50[i]) for i in range(offset*1, n, sub)]]
    # q3 = [[(nl3_q50[i] - nl3_q25[i]) for i in range(offset*2, n, sub)], [(nl3_q75[i] - nl3_q50[i]) for i in range(offset*2, n, sub)]]
    # q4 = [[(nl4_q50[i] - nl4_q25[i]) for i in range(15, n, sub)], [(nl4_q75[i] - nl4_q50[i]) for i in range(15, n, sub)]]

    i_q1 = [[(nl1_i50[i] - nl1_i25[i]) for i in range(0, n, sub)], [(nl1_i75[i] - nl1_i50[i]) for i in range(0, n, sub)]]
    i_q2 = [[(nl2_i50[i] - nl2_i25[i]) for i in range(offset*1, n, sub)], [(nl2_i75[i] - nl2_i50[i]) for i in range(offset*1, n, sub)]]

    r_outlower1 = [nl1_r5[i] for i in range(0, n, sub)]
    r_outlower2 = [nl2_r5[i] for i in range(offset*1, n, sub)]
    # outlower3 = [nl3_q5[i] for i in range(offset*2, n, sub)] 
    # outlower4 = [nl4_q5[i] for i in range(15, n, sub)]
     
    i_outlower1 = [nl1_i5[i] for i in range(0, n, sub)]
    i_outlower2 = [nl2_i5[i] for i in range(offset*1, n, sub)] 

    r_outupper1 = [nl1_r95[i] for i in range(0, n, sub)]
    r_outupper2 = [nl2_r95[i] for i in range(offset*1, n, sub)]
    # outupper3 = [nl3_q95[i] for i in range(offset*2, n, sub)] 
    # outupper4 = [nl4_q95[i] for i in range(15, n, sub)]
    
    i_outupper1 = [nl1_i95[i] for i in range(0, n, sub)]
    i_outupper2 = [nl2_i95[i] for i in range(offset*1, n, sub)] 

    batches1 = [x for x in range(0, n, sub)]
    batches2 = [x for x in range(offset*1, n, sub)]
    # batches3 = [x for x in range(offset*2, n, sub)]
    # batches4 = [x for x in range(15, n, sub)]

    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].errorbar(batches1, r_med1, yerr=r_q1, color='b')
    axes[0].errorbar(batches2, r_med2, yerr=r_q2, color='g')
    # plt.errorbar(batches3, med3, yerr=q3, color='r')
    # plt.errorbar(batches4, med4, yerr=q4, color='c')

    axes[0].scatter(batches1, r_outlower1, color='b')
    axes[0].scatter(batches2, r_outlower2, color='g')
    # plt.scatter(batches3, outlower3, color='r')
    # plt.scatter(batches4, outlower4, color='c')

    axes[0].scatter(batches1, r_outupper1, color='b')
    axes[0].scatter(batches2, r_outupper2, color='g')
    # plt.scatter(batches3, outupper3, color='r')
    # plt.scatter(batches4, outupper4, color='c')

    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Re(f(x))')
    axes[0].legend(legend, loc='upper right')

    axes[1].errorbar(batches1, i_med1, yerr=i_q1, color='b')
    axes[1].errorbar(batches2, i_med2, yerr=i_q2, color='g')

    axes[1].scatter(batches1, i_outlower1, color='b')
    axes[1].scatter(batches2, i_outlower2, color='g')

    axes[1].scatter(batches1, i_outupper1, color='b')
    axes[1].scatter(batches2, i_outupper2, color='g')

    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Im(f(x))')
    # axes[1].legend(legend, loc='upper right')

    plt.show()
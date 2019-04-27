'''Author: Ben Steel
Date: 19/03/19'''

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import numpy as np 

if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_8Gamma_lr01_bias4b_CoherentNet_8_8f5_8p2_128_semeionDataset_NLdata_try0.pkl'

    params, cost, cost_val, nl1_p, nl2_p, nl3_p, nl4_p, final_layer = pickle.load(open(save_path, 'rb'))

    [nl1_r5, nl1_r25, nl1_r50, nl1_r75, nl1_r95, nl1_i5, nl1_i25, nl1_i50, nl1_i75, nl1_i95] = nl1_p
    [nl2_r5, nl2_r25, nl2_r50, nl2_r75, nl2_r95, nl2_i5, nl2_i25, nl2_i50, nl2_i75, nl2_i95] = nl2_p
    [nl3_r5, nl3_r25, nl3_r50, nl3_r75, nl3_r95, nl3_i5, nl3_i25, nl3_i50, nl3_i75, nl3_i95] = nl3_p
    [nl4_r5, nl4_r25, nl4_r50, nl4_r75, nl4_r95, nl4_i5, nl4_i25, nl4_i50, nl4_i75, nl4_i95] = nl4_p

    # legend = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]
    legend = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]

    m = 30
    k = 4
    n = m*k
    num_points = 20
    sub = n // num_points
    offset = sub // 4
    
    r_med1 = [nl1_r50[i] for i in range(0, n, sub)]
    r_med2 = [nl2_r50[i] for i in range(offset*1, n, sub)]
    r_med3 = [nl3_r50[i] for i in range(offset*2, n, sub)]
    r_med4 = [nl4_r50[i] for i in range(offset*3, n, sub)]

    i_med1 = [nl1_i50[i] for i in range(0, n, sub)]
    i_med2 = [nl2_i50[i] for i in range(offset*1, n, sub)]
    i_med3 = [nl3_i50[i] for i in range(offset*2, n, sub)]
    i_med4 = [nl4_i50[i] for i in range(offset*3, n, sub)]

    r_q1 = [[(nl1_r50[i] - nl1_r25[i]) for i in range(0, n, sub)], [(nl1_r75[i] - nl1_r50[i]) for i in range(0, n, sub)]]
    r_q2 = [[(nl2_r50[i] - nl2_r25[i]) for i in range(offset*1, n, sub)], [(nl2_r75[i] - nl2_r50[i]) for i in range(offset*1, n, sub)]]
    r_q3 = [[(nl3_r50[i] - nl3_r25[i]) for i in range(offset*2, n, sub)], [(nl3_r75[i] - nl3_r50[i]) for i in range(offset*2, n, sub)]]
    r_q4 = [[(nl4_r50[i] - nl4_r25[i]) for i in range(offset*3, n, sub)], [(nl4_r75[i] - nl4_r50[i]) for i in range(offset*3, n, sub)]]

    i_q1 = [[(nl1_i50[i] - nl1_i25[i]) for i in range(0, n, sub)], [(nl1_i75[i] - nl1_i50[i]) for i in range(0, n, sub)]]
    i_q2 = [[(nl2_i50[i] - nl2_i25[i]) for i in range(offset*1, n, sub)], [(nl2_i75[i] - nl2_i50[i]) for i in range(offset*1, n, sub)]]
    i_q3 = [[(nl3_i50[i] - nl3_i25[i]) for i in range(offset*2, n, sub)], [(nl3_i75[i] - nl3_i50[i]) for i in range(offset*2, n, sub)]]
    i_q4 = [[(nl4_i50[i] - nl4_i25[i]) for i in range(offset*3, n, sub)], [(nl4_i75[i] - nl4_i50[i]) for i in range(offset*3, n, sub)]]

    # r_qlower1 = [nl1_r25[i] for i in range(0, n, sub)]
    # r_qlower2 = [nl2_r25[i] for i in range(offset*1, n, sub)]
    # r_qlower3 = [nl3_r25[i] for i in range(offset*2, n, sub)]
    # r_qlower4 = [nl4_r25[i] for i in range(offset*3, n, sub)]
     
    # i_qlower1 = [nl1_i25[i] for i in range(0, n, sub)]
    # i_qlower2 = [nl2_i25[i] for i in range(offset*1, n, sub)] 
    # i_qlower3 = [nl3_i25[i] for i in range(offset*2, n, sub)] 
    # i_qlower4 = [nl4_i25[i] for i in range(offset*3, n, sub)] 

    # r_qupper1 = [nl1_r75[i] for i in range(0, n, sub)]
    # r_qupper2 = [nl2_r75[i] for i in range(offset*1, n, sub)]
    # r_qupper3 = [nl3_r75[i] for i in range(offset*2, n, sub)]
    # r_qupper4 = [nl4_r75[i] for i in range(offset*3, n, sub)]
    
    # i_qupper1 = [nl1_i75[i] for i in range(0, n, sub)]
    # i_qupper2 = [nl2_i75[i] for i in range(offset*1, n, sub)] 
    # i_qupper3 = [nl3_i75[i] for i in range(offset*2, n, sub)] 
    # i_qupper4 = [nl4_i75[i] for i in range(offset*3, n, sub)] 

    r_outlower1 = [nl1_r5[i] for i in range(n)]
    r_outlower2 = [nl2_r5[i] for i in range(n)]
    r_outlower3 = [nl3_r5[i] for i in range(n)]
    r_outlower4 = [nl4_r5[i] for i in range(n)]
     
    i_outlower1 = [nl1_i5[i] for i in range(n)]
    i_outlower2 = [nl2_i5[i] for i in range(n)] 
    i_outlower3 = [nl3_i5[i] for i in range(n)] 
    i_outlower4 = [nl4_i5[i] for i in range(n)] 

    r_outupper1 = [nl1_r95[i] for i in range(n)]
    r_outupper2 = [nl2_r95[i] for i in range(n)]
    r_outupper3 = [nl3_r95[i] for i in range(n)]
    r_outupper4 = [nl4_r95[i] for i in range(n)]
    
    i_outupper1 = [nl1_i95[i] for i in range(n)]
    i_outupper2 = [nl2_i95[i] for i in range(n)]
    i_outupper3 = [nl3_i95[i] for i in range(n)]
    i_outupper4 = [nl4_i95[i] for i in range(n)]

    batches1 = [x/k for x in range(0, n, sub)]
    batches2 = [x/k for x in range(offset*1, n, sub)]
    batches3 = [x/k for x in range(offset*2, n, sub)]
    batches4 = [x/k for x in range(offset*3, n, sub)]

    # batches1 = np.linspace(0, m, num_points + 2)
    # batches2 = np.linspace(offset*1*m/n, m + offset*1*m/n, num_points + 2)
    # batches3 = np.linspace(offset*2*m/n, m + offset*2*m/n, num_points + 2)
    # batches4 = np.linspace(offset*3*m/n, m + offset*3*m/n, num_points + 1)

    out_batches = np.linspace(0, m, n)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 10))

    axes[0].errorbar(batches1, r_med1, yerr=r_q1, color='b')
    axes[0].errorbar(batches2, r_med2, yerr=r_q2, color='g')
    axes[0].errorbar(batches3, r_med3, yerr=r_q3, color='r')
    axes[0].errorbar(batches4, r_med4, yerr=r_q4, color='c')

    # axes[0].plot(batches1, r_med1, color='b')
    # axes[0].plot(batches2, r_med2, color='g')
    # axes[0].plot(batches3, r_med3, color='r')
    # axes[0].plot(batches4, r_med4, color='c')

    # axes[0].plot(batches1, r_qlower1, color='b')
    # axes[0].plot(batches2, r_qlower2, color='g')
    # axes[0].plot(batches3, r_qlower3, color='r')
    # axes[0].plot(batches4, r_qlower4, color='c')

    # axes[0].plot(batches1, r_qupper1, color='b')
    # axes[0].plot(batches2, r_qupper2, color='g')
    # axes[0].plot(batches3, r_qupper3, color='r')
    # axes[0].plot(batches4, r_qupper4, color='c')

    axes[0].plot(out_batches, r_outlower1, color='b')
    axes[0].plot(out_batches, r_outlower2, color='g')
    axes[0].plot(out_batches, r_outlower3, color='r')
    axes[0].plot(out_batches, r_outlower4, color='c')

    axes[0].plot(out_batches, r_outupper1, color='b')
    axes[0].plot(out_batches, r_outupper2, color='g')
    axes[0].plot(out_batches, r_outupper3, color='r')
    axes[0].plot(out_batches, r_outupper4, color='c')

    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel(r'$\Re(f(x))$')
    start, end = -2, 2.01
    step = 0.4
    axes[0].set_ylim(bottom=start, top=end)
    axes[0].yaxis.set_ticks(np.arange(start, end, step))
    axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    axes[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    axes[0].legend(legend, loc='upper left')

    axes[1].errorbar(batches1, i_med1, yerr=i_q1, color='b')
    axes[1].errorbar(batches2, i_med2, yerr=i_q2, color='g')
    axes[1].errorbar(batches3, i_med3, yerr=i_q3, color='r')
    axes[1].errorbar(batches4, i_med4, yerr=i_q4, color='c')

    # axes[1].plot(batches1, i_med1, color='b')
    # axes[1].plot(batches2, i_med2, color='g')
    # axes[1].plot(batches3, i_med3, color='r')
    # axes[1].plot(batches4, i_med4, color='c')

    # axes[1].plot(batches1, i_qlower1, color='b')
    # axes[1].plot(batches2, i_qlower2, color='g')
    # axes[1].plot(batches3, i_qlower3, color='r')
    # axes[1].plot(batches4, i_qlower4, color='c')

    # axes[1].plot(batches1, i_qupper1, color='b')
    # axes[1].plot(batches2, i_qupper2, color='g')
    # axes[1].plot(batches3, i_qupper3, color='r')
    # axes[1].plot(batches4, i_qupper4, color='c')

    axes[1].plot(out_batches, i_outlower1, color='b')
    axes[1].plot(out_batches, i_outlower2, color='g')
    axes[1].plot(out_batches, i_outlower3, color='r')
    axes[1].plot(out_batches, i_outlower4, color='c')

    axes[1].plot(out_batches, i_outupper1, color='b')
    axes[1].plot(out_batches, i_outupper2, color='g')
    axes[1].plot(out_batches, i_outupper3, color='r')
    axes[1].plot(out_batches, i_outupper4, color='c')

    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel(r'$\Im(f(x))$')
    axes[1].set_ylim(bottom=start, top=end)
    axes[1].yaxis.set_ticks(np.arange(start, end, step))
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    axes[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

    axes[2].plot(out_batches, cost[:n])
    axes[2].plot(cost_val[:m])
    axes[2].set_xlabel('Epochs')
    axes[2].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    axes[2].set_ylabel('Cost')
    axes[2].legend(['Training Loss', 'Validation Loss'], loc='upper right')

    plt.show()
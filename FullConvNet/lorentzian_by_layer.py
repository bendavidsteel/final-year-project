import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    
    save_path = 'adamGD_SoftmaxCross_2overpiGamma_FullNet128_SimpleDigits_NLquartiledata'

    params, cost, layer_q5, layer_q25, layer_q50, layer_q75, layer_q95, final_layer = pickle.load(open(save_path, 'rb'))

    # [nl1_q5, nl2_q5, nl3_q5, nl4_q5, nl5_q5] = layer_q5
    # [nl1_q25, nl2_q25, nl3_q25, nl4_q25, nl5_q25] = layer_q25
    # [nl1_q50, nl2_q50, nl3_q50, nl4_q50, nl5_q50] = layer_q50
    # [nl1_q75, nl2_q75, nl3_q75, nl4_q75, nl5_q75] = layer_q75
    # [nl1_q95, nl2_q95, nl3_q95, nl4_q95, nl5_q95] = layer_q95

    [nl1_q5, nl2_q5, nl3_q5, nl4_q5] = layer_q5
    [nl1_q25, nl2_q25, nl3_q25, nl4_q25] = layer_q25
    [nl1_q50, nl2_q50, nl3_q50, nl4_q50] = layer_q50
    [nl1_q75, nl2_q75, nl3_q75, nl4_q75] = layer_q75
    [nl1_q95, nl2_q95, nl3_q95, nl4_q95] = layer_q95

    legend = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]

    sub = 25
    n = len(nl1_q50)
    
    med1 = [nl1_q50[i] for i in range(0, n, sub)]
    med2 = [nl2_q50[i] for i in range(5, n, sub)]
    med3 = [nl3_q50[i] for i in range(10, n, sub)]
    med4 = [nl4_q50[i] for i in range(15, n, sub)]
    # med5 = [nl5_q50[i] for i in range(20, n, sub)]

    q1 = [[(nl1_q50[i] - nl1_q25[i]) for i in range(0, n, sub)], [(nl1_q75[i] - nl1_q50[i]) for i in range(0, n, sub)]]
    q2 = [[(nl2_q50[i] - nl2_q25[i]) for i in range(5, n, sub)], [(nl2_q75[i] - nl2_q50[i]) for i in range(5, n, sub)]]
    q3 = [[(nl3_q50[i] - nl3_q25[i]) for i in range(10, n, sub)], [(nl3_q75[i] - nl3_q50[i]) for i in range(10, n, sub)]]
    q4 = [[(nl4_q50[i] - nl4_q25[i]) for i in range(15, n, sub)], [(nl4_q75[i] - nl4_q50[i]) for i in range(15, n, sub)]]
    # q5 = [[(nl5_q50[i] - nl5_q25[i]) for i in range(20, n, sub)], [(nl5_q75[i] - nl5_q50[i]) for i in range(20, n, sub)]]

    outlower1 = [nl1_q5[i] for i in range(0, n, sub)]
    outlower2 = [nl2_q5[i] for i in range(5, n, sub)]
    outlower3 = [nl3_q5[i] for i in range(10, n, sub)] 
    outlower4 = [nl4_q5[i] for i in range(15, n, sub)]
    # outlower5 = [nl5_q5[i] for i in range(20, n, sub)]

    outupper1 = [nl1_q95[i] for i in range(0, n, sub)]
    outupper2 = [nl2_q95[i] for i in range(5, n, sub)]
    outupper3 = [nl3_q95[i] for i in range(10, n, sub)] 
    outupper4 = [nl4_q95[i] for i in range(15, n, sub)] 
    # outupper5 = [nl5_q95[i] for i in range(20, n, sub)] 

    batches1 = [x for x in range(0, n, sub)]
    batches2 = [x for x in range(5, n, sub)]
    batches3 = [x for x in range(10, n, sub)]
    batches4 = [x for x in range(15, n, sub)]
    # batches5 = [x for x in range(20, n, sub)]

    plt.errorbar(batches1, med1, yerr=q1, color='b')
    plt.errorbar(batches2, med2, yerr=q2, color='g')
    plt.errorbar(batches3, med3, yerr=q3, color='r')
    plt.errorbar(batches4, med4, yerr=q4, color='c')
    # plt.errorbar(batches5, med5, yerr=q5, color='m')

    plt.scatter(batches1, outlower1, color='b')
    plt.scatter(batches2, outlower2, color='g')
    plt.scatter(batches3, outlower3, color='r')
    plt.scatter(batches4, outlower4, color='c')
    # plt.scatter(batches5, outlower5, color='m')

    plt.scatter(batches1, outupper1, color='b')
    plt.scatter(batches2, outupper2, color='g')
    plt.scatter(batches3, outupper3, color='r')
    plt.scatter(batches4, outupper4, color='c')
    # plt.scatter(batches5, outupper5, color='m')

    plt.xlabel('Batch Updates')
    plt.ylabel('Activation Value')
    plt.legend(legend, loc='upper right')
    plt.show()
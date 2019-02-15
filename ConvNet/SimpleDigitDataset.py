import numpy as np


def generateDataset():

    zero0 = np.array([[0,0,0,0,0,0,0,0],
                      [0,0,0,1,1,0,0,0],
                      [0,0,1,0,0,1,0,0],
                      [0,1,0,0,0,0,1,0],
                      [0,1,0,0,0,0,1,0],
                      [0,1,0,0,0,0,1,0],
                      [0,0,1,0,0,1,0,0],
                      [0,0,0,1,1,0,0,0]])

    zero1 = np.array([[0,0,0,0,0,0,0,0],
                      [0,0,1,1,1,0,0,0],
                      [0,1,0,0,0,1,0,0],
                      [0,1,0,0,0,1,0,0],
                      [0,1,0,0,0,1,0,0],
                      [0,1,0,0,0,1,0,0],
                      [0,0,1,1,1,0,0,0],
                      [0,0,0,0,0,0,0,0]])

    zero2 = np.array([[0,0,0,0,1,0,0,0],
                      [0,0,0,1,0,1,0,0],
                      [0,0,1,0,0,0,1,0],
                      [0,0,1,0,0,0,1,0],
                      [0,0,1,0,0,0,1,0],
                      [0,0,0,1,0,1,0,0],
                      [0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,0,0]])

    zero3 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,0,0,0,0,0,0]])

    zero4 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    one0 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    one1 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,1,1,1,1,1,0],
                    [0,0,0,0,0,0,0,0]])

    one2 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,1,1,0,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    one3 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,0,0,0,0,0,0]])

    one4 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    two0 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,1,1,1,1,0,0],
                    [0,0,0,0,0,0,0,0]])

    two1 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,1,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,1,1,1,1,1,0,0],
                    [0,0,0,0,0,0,0,0]])

    two2 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,1,0,0],
                    [0,0,0,1,0,0,1,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,1,1,1,0],
                    [0,0,0,0,0,0,0,0]])

    two3 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,1,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,1,1,0,0,0,0],
                    [0,1,1,1,1,1,0,0],
                    [0,0,0,0,0,0,0,0]])

    two4 = np.array([[0,0,0,0,1,1,0,0],
                    [0,0,0,1,0,0,1,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,1,1,1,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    three0 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,1,0,0],
                    [0,0,0,1,0,0,1,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,1,0,0,1,0],
                    [0,0,0,0,1,1,0,0]])

    three1 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,0,1,1,1,0,0]])

    three2 = np.array([[0,0,1,1,1,0,0,0],
                    [0,1,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,1,0,0,0,1,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    three3 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0]])

    three4 = np.array([[0,0,0,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    four0 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,1,0,1,0,0,0],
                    [0,1,1,1,1,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    four1 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,1,0,0],
                    [0,0,0,1,0,1,0,0],
                    [0,0,1,1,1,1,1,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0]])

    four2 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,1,0,0],
                    [0,0,0,1,0,1,0,0],
                    [0,0,0,1,0,1,0,0],
                    [0,0,1,1,1,1,1,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0]])

    four3 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,1,0,0],
                    [0,0,0,1,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,1,1,1,1,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0]])

    four4 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,1,0,1,0,0,0],
                    [0,0,1,0,1,0,0,0],
                    [0,1,0,0,1,0,0,0],
                    [0,1,1,1,1,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    five0 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,1,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0]])

    five1 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,1,1,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,1,0,0,1,0],
                    [0,0,0,0,1,1,0,0]])

    five2 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,1,1,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,1,0,0,1,0],
                    [0,0,0,0,1,1,0,0]])

    five3 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,1,1,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,1,1,1,1,0,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,0,0,0,0,0,0]])

    five4 = np.array([[0,0,1,1,1,1,1,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,1,1,1,1,0,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    six0 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0]])

    six1 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0]])

    six2 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    six3 = np.array([[0,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    six4 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,1,1,1,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    seven0 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    seven1 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    seven2 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,1,1,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    seven3 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,1,1,0],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    seven4 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,1,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0]])

    eight0 = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,0,1,1,1,0,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,1,0,0,0,1,0],
                    [0,0,0,1,1,1,0,0]])

    eight1 = np.array([[0,0,0,1,1,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,1,1,0,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,1,0,0,0,1,0],
                       [0,0,0,1,1,1,0,0],
                       [0,0,0,0,0,0,0,0]])

    eight2 = np.array([[0,0,0,1,1,0,0,0],
                       [0,0,1,0,0,1,0,0],
                       [0,0,1,0,0,1,0,0],
                       [0,0,0,1,1,0,0,0],
                       [0,0,1,0,0,1,0,0],
                       [0,0,1,0,0,1,0,0],
                       [0,0,0,1,1,0,0,0],
                       [0,0,0,0,0,0,0,0]])

    eight3 = np.array([[0,0,0,0,0,0,0,0],
                       [0,0,0,0,1,1,0,0],
                       [0,0,0,1,0,0,1,0],
                       [0,0,0,1,0,0,1,0],
                       [0,0,0,0,1,1,0,0],
                       [0,0,0,1,0,0,1,0],
                       [0,0,0,1,0,0,1,0],
                       [0,0,0,0,1,1,0,0]])

    eight4 = np.array([[0,0,1,1,1,0,0,0],
                       [0,1,0,0,0,1,0,0],
                       [0,1,0,0,0,1,0,0],
                       [0,0,1,1,1,0,0,0],
                       [0,1,0,0,0,1,0,0],
                       [0,1,0,0,0,1,0,0],
                       [0,0,1,1,1,0,0,0],
                       [0,0,0,0,0,0,0,0]])

    nine0 = np.array([[0,0,0,1,1,0,0,0],
                      [0,0,1,0,0,1,0,0],
                      [0,0,1,0,0,1,0,0],
                      [0,0,0,1,1,1,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,1,1,0,0,0],
                      [0,0,0,0,0,0,0,0]])

    nine1 = np.array([[0,0,0,1,1,0,0,0],
                      [0,0,1,0,0,1,0,0],
                      [0,0,1,0,0,1,0,0],
                      [0,0,0,1,1,1,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,0,0]])

    nine2 = np.array([[0,0,0,1,1,0,0,0],
                      [0,0,1,0,0,1,0,0],
                      [0,0,1,0,0,1,0,0],
                      [0,0,0,1,1,1,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,0,0]])

    nine3 = np.array([[0,0,0,0,0,0,0,0],
                      [0,0,1,1,1,0,0,0],
                      [0,1,0,0,0,1,0,0],
                      [0,1,0,0,0,1,0,0],
                      [0,0,1,1,1,1,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,1,0,0]])

    nine4 = np.array([[0,0,0,0,0,0,0,0],
                      [0,0,0,1,1,1,0,0],
                      [0,0,1,0,0,0,1,0],
                      [0,0,1,0,0,0,1,0],
                      [0,0,0,1,1,1,1,0],
                      [0,0,0,0,0,0,1,0],
                      [0,0,0,0,0,0,1,0],
                      [0,0,0,0,0,0,1,0]])

    dataset = np.array([zero0, zero1, zero2, zero3, zero4,
                        one0, one1, one2, one3, one4,
                        two0, two1, two2, two3, two4,
                        three0, three1, three2, three3, three4,
                        four0, four1, four2, four3, four4,
                        five0, five1, five2, five3, five4,
                        six0, six1, six2, six3, six4,
                        seven0, seven1, seven2, seven3, seven4,
                        eight0, eight1, eight2, eight3, eight4,
                        nine0, nine1, nine2, nine3, nine4], dtype = np.float32)

    num_cats = 10
    num_for_each = 5

    labels = np.zeros((num_cats * num_for_each, 1))

    for i in range(num_cats):
        for j in range(num_for_each):
            labels[(i*4) + j] = i

    return dataset.reshape((num_cats * num_for_each, 8*8)), np.asarray(labels)

generateDataset()
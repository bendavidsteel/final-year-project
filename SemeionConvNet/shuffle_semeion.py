'''Author : Ben Steel
Date : 08/02/19'''

import numpy as np 

with open('semeion.data', 'r') as f:

    lines = []

    for line in f:
        lines.append(line)

lines = np.asarray(lines)

np.random.shuffle(lines)

with open('semeion_shuffled.data', 'w') as f:
    for line in lines:
        f.write(line)
import numpy as np 

with open('heart.csv', 'r') as f:

    lines = []

    first = True
    first_line = ""

    for line in f:
        if first:
            first_line = line
            first = False
        else:
            lines.append(line)

lines = np.asarray(lines)

np.random.shuffle(lines)

with open('heart_shuffled.csv', 'w') as f:

    f.write(first_line)

    for line in lines:
        f.write(line)
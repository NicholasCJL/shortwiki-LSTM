import numpy as np

# obtain dataset and split into character-wise array
def get_data(dir):
    char_data = []
    with open(dir, 'r') as file:
        for lines in file:
            char_data.extend(list(lines))
    return char_data

# split dataset into samples with n context steps and 1 answer
def split_sequence(sequence, n):
    x, y = [], []
    for i in range(len(sequence)):
        # check if reached the end
        end_ix = i + n
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.asarray(x), np.asarray(y)
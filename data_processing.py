import numpy as np

class Translator:
    def __init__(self, sequence):
        self.sequence = sequence
        self.vocab, self.c2i, self.i2c = self.get_vocab()

    def char_to_int(self, sequence):
        int_sequence = [self.c2i[c] for c in sequence]
        return int_sequence

    def int_to_char(self, sequence):
        char_sequence = [self.i2c[i] for i in sequence]
        return char_sequence

    # get vocabulary and form mapping
    def get_vocab(self):
        chars = sorted(list(set(self.sequence)))
        translator_c2i = {c: i for (i, c) in enumerate(chars)}
        translator_i2c = {i: c for (i, c) in enumerate(chars)}
        return chars, translator_c2i, translator_i2c

# obtain dataset and split into character-wise array
def get_data(dir):
    char_data = []
    with open(dir, 'r', encoding='utf8') as file:
        for line in file:
            char_data.extend(list(line))
    return char_data

# split dataset into samples with n context steps and 1 answer
def split_sequence(sequence, n, alert=999999999999999999):
    x, y = [], []
    i = 1
    for i in range(len(sequence)):
        # check if reached the end
        end_ix = i + n
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
        if i % alert == 0:
            print(f"Number of samples processed: {i}")
    print(f"Number of samples: {len(x)}")
    return np.asarray(x), np.asarray(y)


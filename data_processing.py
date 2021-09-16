import numpy as np
import random
import pickle
import keras
from keras.utils import np_utils

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

def save_batches(data, folder, prefix, batch_size, filesize=2000): # one-hot encoding to be done on batch generation
    path = f'{folder}/{prefix}'
    curr_index = 0
    batches = []
    seq_length = len(data[0])
    chunk_size = int((filesize / 7) / (batch_size * seq_length / (1600))) # baseline of 7 kB
    while True:
            if curr_index + batch_size <= len(data):
                # add batch to list of batches
                new_batch = data[curr_index:curr_index+batch_size]
                np_batch = np.reshape(new_batch, (batch_size, seq_length, 1))
                batches.append(np_batch)
                curr_index += batch_size
                if curr_index == len(data):
                    break
            else:
                # randomly sample from all the data to fill up final batch
                to_sample = batch_size - (len(data) - curr_index)
                last_batch = data[curr_index:]
                to_add = np.asarray(random.choices(data, k=to_sample))
                last_batch = np.append(last_batch, to_add, 0)
                np_batch = np.reshape(last_batch, (batch_size, seq_length, 1))
                batches.append(np_batch)
                break

    print(f'Num batches for {prefix}: {len(batches)}')
    # group batches into chunks to save into one file
    curr_index = 0
    while True:
        if curr_index + chunk_size <= len(batches):
            curr_chunk = batches[curr_index:curr_index + chunk_size]
            with open(f'{path}_{batch_size}_{curr_index}_{curr_index+chunk_size-1}.pkl', 'wb') as file:
                pickle.dump([len(curr_chunk), curr_chunk], file)
            print(f'{len(curr_chunk)} batches {curr_index} to {curr_index+chunk_size-1} '
                  f'of batch size {batch_size} saved to '
                  f'{path}_{batch_size}_{curr_index}_{curr_index+chunk_size-1}.pkl')
            curr_index += chunk_size
            if curr_index == len(batches):
                break
        else:
            curr_chunk = batches[curr_index:]
            with open(f'{path}_{batch_size}_{curr_index}_{curr_index+chunk_size-1}.pkl', 'wb') as file:
                pickle.dump([len(curr_chunk), curr_chunk], file)
            print(f'{len(curr_chunk)} batches {curr_index} to {curr_index + chunk_size - 1} '
                  f'of batch size {batch_size} saved to '
                  f'{path}_{batch_size}_{curr_index}_{curr_index + chunk_size - 1}.pkl')
            break

    return




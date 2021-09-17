import os
import numpy as np
import random
import pickle
from tensorflow.keras.utils import Sequence
from keras.utils import np_utils

# generator for batch training
class BatchGenerator(Sequence):
    def __init__(self, batch_path, translator, one_hot, shuffle=True):
        self.files = os.listdir(batch_path)
        self.shuffle = shuffle
        self.one_hot = one_hot

        if self.one_hot:
            self.translator = translator # for one-hot encoding
            self.generate_one_hot()

        # total number of batches
        with open(f'{batch_path}/len.pkl', 'rb') as file:
            self.length = pickle.load(file)
        self.files.remove('len.pkl') # remove length file from chunk list

        if self.shuffle:
            random.shuffle(self.files)

        # load up first chunk
        with open(f'{batch_path}/{self.files[0]}', 'rb') as file:
            self.curr_chunk_size, self.chunk = pickle.load(file)

        self.chunk_counter = 0 # counter for chunk number
        self.batch_counter = 0 # counter for batch number in chunk

    def generate_one_hot(self): # generate a one hot encoding on the vocab
        vocab = self.translator.vocab
        self.vocab_length = len(vocab)
        self.one_hot_dict = {}
        for i in range(self.vocab_length):
            curr_encoding = np.zeros(self.vocab_length)
            curr_encoding[i] = 1.0
            self.one_hot_dict[i] = curr_encoding

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.batch_counter < self.curr_chunk_size: # load next batch in chunk
            curr_batch = self.chunk[self.batch_counter]
            self.batch_counter += 1
            batch_length = curr_batch.shape[0]
            # convert to one-hot encoding, only implemented for output
            # (batch_size, 1, 1) -> (batch_size, 1, one-hot dims)
            if self.one_hot:
                new_batch = []
                for i in range(batch_length)
                    one_hot_out = self.one_hot_dict[curr_batch[i][0][0]]
                    new_batch.append(one_hot_out)
                new_batch_np = np.asarray(new_batch)
                new_batch_np = new_batch_np.reshape((batch_length, 1, self.vocab_length))
                curr_batch = new_batch_np




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

# filesize - number of batches in file
# one-hot encoding to be done on batch generation
def save_batches(data_x, data_y, folder, prefix, batch_size, filesize=1500):
    path = f'{folder}/{prefix}'
    curr_index = 0
    batches = []
    x_seq_length = len(data_x[0])
    y_seq_length = 1
    chunk_size = filesize
    while True:
            if curr_index + batch_size <= len(data_x):
                # add batch to list of batches
                new_batch_x = data_x[curr_index:curr_index+batch_size]
                np_batch_x = np.reshape(new_batch_x, (batch_size, x_seq_length, 1))
                new_batch_y = data_y[curr_index:curr_index + batch_size]
                np_batch_y = np.reshape(new_batch_y, (batch_size, y_seq_length, 1))
                batches.append([np_batch_x, np_batch_y])
                curr_index += batch_size
                if curr_index == len(data_x):
                    break
            else:
                # randomly sample from all the data to fill up final batch
                to_sample_x = batch_size - (len(data_x) - curr_index)
                last_batch_x = data_x[curr_index:]
                rand_x = random.sample(range(len(data_x)), to_sample_x)
                to_add_x = np.asarray([data_x[i] for i in rand_x])
                last_batch_x = np.append(last_batch_x, to_add_x, 0)
                np_batch_x = np.reshape(last_batch_x, (batch_size, x_seq_length, 1))
                last_batch_y = data_y[curr_index:]
                to_add_y = np.asarray([data_y[i] for i in rand_x])
                last_batch_y = np.append(last_batch_y, to_add_y, 0)
                np_batch_y = np.reshape(last_batch_y, (batch_size, y_seq_length, 1))
                batches.append([np_batch_x, np_batch_y])
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
            with open(f'{path}_{batch_size}_{curr_index}_{curr_index+len(curr_chunk)-1}.pkl', 'wb') as file:
                pickle.dump([len(curr_chunk), curr_chunk], file)
            print(f'{len(curr_chunk)} batches {curr_index} to {curr_index + len(curr_chunk) - 1} '
                  f'of batch size {batch_size} saved to '
                  f'{path}_{batch_size}_{curr_index}_{curr_index+len(curr_chunk)-1}.pkl')
            break

    # save log file of total number of batches
    with open(f'{folder}/len.pkl', 'wb') as file:
        pickle.dump(len(batches), file)

    return




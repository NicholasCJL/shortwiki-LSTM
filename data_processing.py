import os
import numpy as np
import random
import pickle
from tensorflow.keras.utils import Sequence
from keras.utils import np_utils

# generator for batch training
class BatchGenerator(Sequence):
    def __init__(self, batch_path, translator, shuffle=True):
        self.batch_path = batch_path
        self.files = os.listdir(self.batch_path)
        self.files.remove('len.pkl')
        self.num_chunks = len(self.files)
        self.shuffle = shuffle

        # generate one-hot encoding
        self.translator = translator
        self.generate_one_hot()

        # total number of batches
        with open(f'{batch_path}/len.pkl', 'rb') as file:
            self.length = pickle.load(file)

        if self.shuffle:
            random.shuffle(self.files)

        self.generate_indices()

    # generate a one hot encoding on the vocab
    def generate_one_hot(self):
        vocab = self.translator.vocab
        self.vocab_length = len(vocab)
        self.one_hot_dict = {}
        self.reverse_one_hot = {}
        for i in range(self.vocab_length):
            curr_encoding = np.zeros(self.vocab_length)
            curr_encoding[i] = 1.0
            self.one_hot_dict[i] = curr_encoding
            self.reverse_one_hot[curr_encoding] = [i]

    # generate indices for batches
    def generate_indices(self):
        self.indices = {}
        curr_index = 0
        # loop through each chunk and index it
        for file_num in range(self.num_chunks):
            with open(f'{self.batch_path}/{self.files[file_num]}', 'rb') as file:
                curr_chunk_size, chunk = pickle.load(file)
            for batch_num in range(curr_chunk_size):
                self.indices[curr_index] = (file_num, batch_num)
                curr_index += 1

    # one-hot encoding for batch
    def encode_batch(self, batch):
        batch_size, sequence_length, indv = batch.shape
        if indv != 1:
            # raise error
            print(f'indv = {indv}')
            k = [1]
            print(k[1])

        new_batch = []
        for sequence in range(batch_size):
            new_sequence = []
            for entry in range(sequence):
                key = batch[sequence][entry][0]
                new_sequence.append(self.one_hot_dict[key])
            new_batch.append(new_sequence)

        new_batch = np.asarray(new_batch)
        if (batch_size != new_batch.shape[0]) or (sequence_length != new_batch.shape[1]) \
                or (self.vocab_length != new_batch.shape[2]):
            print("Error: new_batch shape: new_batch.shape")
        return new_batch

    def decode_batch(self, batch):
        batch_size, sequence_length, _ = batch.shape
        new_batch = []

        for sequence in range(batch_size):
            new_sequence = []
            for entry in range(sequence):
                key = batch[sequence][entry]
                new_sequence.append(self.reverse_one_hot[key])
            new_batch.append(new_sequence)

        new_batch = np.asarray(new_batch)
        return new_batch

    # total number of batches for keras to calculate epoch size
    def __len__(self):
        return self.length

    # yield one batch at index
    def __getitem__(self, index):
        # obtain file and batch number corresponding to index
        file_num, batch_num = self.indices[index]

        # obtain batch
        with open(f'{self.batch_path}/{self.files[file_num]}', 'rb') as file:
            _, chunk = pickle.load(file)

        curr_batch = chunk[batch_num]
        X, y = curr_batch[0], curr_batch[1]

        return self.encode_batch(X), self.encode_batch(y)


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




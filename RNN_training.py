import numpy as np
import data_processing as dp
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
import pickle

# remove GPU from tensorflow visibility, set device to CPU (use for small networks)
# tf.config.set_visible_devices([], 'GPU')

# Hyperparameters
BATCH_SIZE = 32
SEQUENCE_LENGTH = 50
LEARNING_RATE = 0.01
DECAY_RATE = 0.97
HIDDEN_LAYER_SIZE = 256
CELLS_SIZE = 2

# train/test split (0-1)
train_portion = 0.8

# prefix for saved files
prefix = "shortwiki"

require_data = False

if require_data:
    # load and process dataset
    dir = "input.txt"
    char_data = dp.get_data(dir) # get data as array of individual characters

    translator = dp.Translator(char_data)

    print(f"Vocabulary: {translator.vocab}")
    print(f"Num chars in vocabulary: {len(translator.c2i)}")

    int_data = translator.char_to_int(char_data) # translate characters into list of ints
    print('Sequence translated.')

    seq_data_context, seq_data_answer =\
        dp.split_sequence(int_data, SEQUENCE_LENGTH, alert=1000000) # split data into sequences of context and answers

    # save processed data
    with open(f'{prefix}_data_context_{SEQUENCE_LENGTH}.pkl', 'wb') as file:
        pickle.dump(seq_data_context, file)
        print('Context saved.')

    with open(f'{prefix}_data_answer_{SEQUENCE_LENGTH}.pkl', 'wb') as file:
        pickle.dump(seq_data_answer, file)
        print('Answers saved.')

    with open(f'{prefix}_translator.pkl', 'wb') as file:
        pickle.dump(translator, file)
        print('Translator saved.')

else:
    # load data from file
    with open(f'{prefix}_data_context_{SEQUENCE_LENGTH}.pkl', 'rb') as file:
        seq_data_context = pickle.load(file)
        print('Context loaded.')

    with open(f'{prefix}_data_answer_{SEQUENCE_LENGTH}.pkl', 'rb') as file:
        seq_data_answer = pickle.load(file)
        print('Answers loaded.')

    with open(f'{prefix}_translator.pkl', 'rb') as file:
        translator = pickle.load(file)
        print('Translator loaded.')


# split dataset into train and test
n_samples = len(seq_data_context)
n_train = int(0.8 * n_samples)
x_train, x_test = seq_data_context[:n_train], seq_data_context[n_train:]
y_train, y_test = seq_data_answer[:n_train], seq_data_answer[n_train:]

print(f'Num samples: {n_samples}')
print(f'Num train: {len(x_train)}')
print(f'Num test: {len(y_test)}')

# # define model
# model = Sequential
# model.add(LSTM(HIDDEN_LAYER_SIZE))

import numpy as np
import data_processing as dp
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
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

# prefix for saved files
prefix = "shortwiki"

require_data = True

if require_data:
    # load and process dataset
    dir = "input.txt"
    char_data = dp.get_data(dir) # get data as array of individual characters
    print(f"Char data obtained with character set: {sorted(list(set(char_data)))}")
    seq_data_context, seq_data_answer =\
        dp.split_sequence(char_data, SEQUENCE_LENGTH) # split data into sequences of context and answers

    # save processed data
    with open(f'{prefix}_data_context_{SEQUENCE_LENGTH}.pkl', 'wb') as file:
        pickle.dump(seq_data_context, file)

    with open(f'{prefix}_data_answer_{SEQUENCE_LENGTH}.pkl', 'wb') as file:
        pickle.dump(seq_data_answer, file)

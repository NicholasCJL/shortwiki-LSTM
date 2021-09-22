import numpy as np
import data_processing as dp
import rnn_model
import tensorflow as tf
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
DROPOUT_RATE = 0.1
HIDDEN_LAYER_SIZE = 256

# train/test split (0-1)
train_portion = 0.8

# prefix for saved files
prefix = "shortwiki"

require_data = False
require_splitting = False

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

elif require_splitting:
    # load data from file
    with open(f'{prefix}_data_context_{SEQUENCE_LENGTH}.pkl', 'rb') as file:
        seq_data_context = pickle.load(file)
        print('Context loaded.')

    with open(f'{prefix}_data_answer_{SEQUENCE_LENGTH}.pkl', 'rb') as file:
        seq_data_answer = pickle.load(file)
        print('Answers loaded.')

    # split dataset into train and test
    n_samples = len(seq_data_context)
    n_train = int(train_portion * n_samples)
    context_train, context_test = seq_data_context[:n_train], seq_data_context[n_train:]
    answer_train, answer_test = seq_data_answer[:n_train], seq_data_answer[n_train:]

    context_length_train, context_length_test = len(context_train), len(context_test)
    answer_length_train, answer_length_test = len(answer_train), len(answer_test)

    print(f'Num samples: {n_samples}')
    print(f'Num train: {context_length_train}')
    print(f'Num test: {answer_length_test}')
    print(f'Verification: {context_length_train + answer_length_test == n_samples}')

    # split train
    dp.save_batches(context_train, answer_train, 'batches/train', 'context_answer', BATCH_SIZE, filesize=1000)
    # split test
    dp.save_batches(context_test, answer_test, 'batches/test', 'context_answer', BATCH_SIZE, filesize=1000)

    # # split context_train
    # dp.save_batches(context_train, 'batches/x_train', 'context_train', BATCH_SIZE, filesize=10000)
    # # split context_test
    # dp.save_batches(context_test, 'batches/x_test', 'context_test', BATCH_SIZE, filesize=10000)
    # # split answer_train
    # answer_train = np.reshape(answer_train, (len(answer_train), 1))
    # dp.save_batches(answer_train, 'batches/y_train', 'answer_train', BATCH_SIZE, filesize=10000)
    # # split answer_test
    # answer_test = np.reshape(answer_test, (len(answer_test), 1))
    # dp.save_batches(answer_test, 'batches/y_test', 'answer_test', BATCH_SIZE, filesize=10000)

with open(f'{prefix}_translator.pkl', 'rb') as file:
    translator = pickle.load(file)
    print('Translator loaded.')

# get model
# in_shape = (SEQUENCE_LENGTH, 1)
in_shape = (SEQUENCE_LENGTH, len(translator.vocab))
print(in_shape)
model = rnn_model.seq_model(in_shape, HIDDEN_LAYER_SIZE,
                            len(translator.vocab), DROPOUT_RATE, LEARNING_RATE)

# checkpoint location
filepath = "model/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
time_history = rnn_model.TimeHistory()
callbacks_list = [checkpoint, time_history]

# fitting
training_batch_generator = dp.BatchGenerator('batches/train', translator)
validation_batch_generator = dp.BatchGenerator('batches/test', translator)
model.fit(training_batch_generator,
          validation_data=validation_batch_generator,
          epochs=200,
          verbose=2,
          callbacks=callbacks_list,
          workers=1,
          shuffle=False)

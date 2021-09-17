from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


def seq_model(in_shape, h_layer_size, out_layer_size, p_dropout, learn_rate, loss_type='categorical_crossentropy', opt=Adam):
    model = Sequential()
    model.add(LSTM(h_layer_size, input_shape=in_shape, return_sequences=True))
    model.add(Dropout(p_dropout))
    model.add(LSTM(h_layer_size))
    model.add(Dropout(p_dropout))
    model.add(Dense(out_layer_size, activation='softmax'))
    model.compile(loss=loss_type, optimizer=opt(learning_rate=learn_rate))
    return model

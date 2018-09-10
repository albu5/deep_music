import pretty_midi
import numpy as np
from keras.models import Sequential
import os
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt

batch_size = 20
skip_size = 5
n_epoch = 30

model = Sequential()
model.add(LSTM(256, input_shape=(batch_size, 128), kernel_initializer='uniform', stateful=True))
model.add(Dense(128, kernel_initializer='uniform'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='logcosh')

train_error_epoch = []
test_error_epoch = []

midi_files_train = []
midi_files_test = []
train_data = []
test_data = []

for file in os.listdir("./../data/beeth_train"):
    if file.endswith(".mid"):
        midi_files_train.append(os.path.join("./../data/beeth_train", file))

for item in midi_files_train:
    midi_data = pretty_midi.PrettyMIDI(item)
    time = midi_data.get_end_time()
    midi_data.time_to_tick(time)
    instrument = midi_data.instruments
    instrument_notes = []
    for instrument in midi_data.instruments:
        print("Instrument: " + pretty_midi.program_to_instrument_name(instrument.program) + " " + str(
            instrument.get_piano_roll().shape))
        if pretty_midi.program_to_instrument_name(instrument.program) == "Acoustic Grand Piano":
            train_data.append(instrument.get_piano_roll())

for file in os.listdir("./../data/beeth_test"):
    if file.endswith(".mid"):
        midi_files_test.append(os.path.join("./../data/beeth_test", file))

for item in midi_files_test:
    midi_data = pretty_midi.PrettyMIDI(item)
    time = midi_data.get_end_time()
    midi_data.time_to_tick(time)
    instrument = midi_data.instruments
    instrument_notes = []
    for instrument in midi_data.instruments:
        print("Instrument: " + pretty_midi.program_to_instrument_name(instrument.program) + " " + str(
            instrument.get_piano_roll().shape))
        if pretty_midi.program_to_instrument_name(instrument.program) == "Acoustic Grand Piano":
            test_data.append(instrument.get_piano_roll())

for epoch in range(n_epoch):
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    counter = 0
    train_error_list = []
    test_error_list = []
    for item in train_data:
        counter += 1
        print("Number of training files processed: %d" % counter)
        for i in range(0, item.shape[1] - batch_size - 2, skip_size):
            x_train.append(item[:, i:batch_size + i])
            y_train.append(item[:, i + batch_size + 1])
            train_error_list.append(model.train_on_batch(np.expand_dims(x_train[-1], axis=0),
                                                         np.expand_dims(y_train[-1], axis=0)))

    counter = 0
    for item in test_data:
        counter += 1
        print("Number of testing files processed: %d" % counter)
        for i in range(0, item.shape[1] - batch_size - 2, skip_size):
            x_test.append(item[:, i:batch_size + i])
            y_test.append(item[:, i + batch_size])
            test_error_list.append(model.evaluate(np.expand_dims(x_test[-1], axis=0),
                                                  np.expand_dims(y_test[-1], axis=0)))
    train_error_epoch.append(np.mean(np.array(train_error_list)))
    test_error_epoch.append(np.mean(np.array(test_error_list)))
    plt.plot(train_error_epoch, 'r')
    plt.plot(test_error_epoch, 'g')
    print("Not returning fuck!")

print("Fucks returned...")
'''
input_layer = Input(shape=(128, batch_size), name='input_layer')
rnn_out = LSTM(units=256, name='LSTM', return_sequences=True )(input_layer)
output = Dense(units=128, activation='linear')(rnn_out)
lstm_model = Model(inputs=[input_layer], outputs=[output])
print(lstm_model.summary())
lstm_model.compile(optimizer=rmsprop(0.001), loss=mean_squared_error)
lstm_model.fit(x_train, y_train, epochs=30)

train_error = lstm_model.evaluate(x_train, y_train)
test_error = lstm_model.evaluate(x_test, y_test)
print("train error: %f, test error: %f" % (train_error, test_error))
'''

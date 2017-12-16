import numpy as np
from keras.layers import GRU, LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, load_model, Model
import timeit


class NNet(object):
    def __init__(self):
        #last_name = ""
        # Parameters:
        self.BATCH_SIZE = 5
        self.TIMESTEPS = 1
        self.EPOCHS = 1
        #self.last_value = 0
        self.model = Sequential()
        self.history = []



    def compile_model(self):
        print("Start Compiling Model")
        t1 = timeit.default_timer()

        np.random.seed(1)

        self.model.add(Dense(units=5, batch_input_shape=(self.BATCH_SIZE, None, 1)))
        #self.model.add(Activation('hard_sigmoid'))
        self.model.add(GRU(units=10, return_sequences=True, stateful=True))
        #self.model.add(Activation('tanh'))
        #self.model.add(Dense(units=1))

        #self.model.add(Dense(units=5))

        #self.model.add(GRU(units=64, return_sequences=True, stateful=True, batch_input_shape=(self.BATCH_SIZE, None, 1), ))# return_sequences=True, stateful=True))
        #self.model.add(Activation('hard_sigmoid'))
        #self.model.add(GRU(units=32, return_sequences=True, stateful=True))
        #self.model.add(Activation('hard_sigmoid'))
        #self.model.add(GRU(units=16, return_sequences=True, stateful=True))
        self.model.add(Dense(units=1))
        #self.model.add(Activation('hard_sigmoid'))


        #self.model.compile(loss="mean_absolute_error", optimizer="rmsprop", metrics=['accuracy'])
        self.model.compile(loss="mean_absolute_error", optimizer="sgd", metrics=['accuracy'])


        t2 = timeit.default_timer()
        np.random.seed()
        print("Model Compiled, took: {}".format(t2 - t1))


    def train_and_predict(self, value):
        # only get last value from window
        #if len(values) > 1:
            # only use last value
        #    values = values[-1]

        # append value to list and create an numpy array
        self.history = np.append(self.history, value)

        # check if there is only one historical value instead of the necessary two values
        if len(self.history) < self.BATCH_SIZE+1:
            return value


        train_x = self.reshape_training_set(self.history[:-1])
        train_y = self.reshape_training_set(self.history[1:])

        #print("train_x: ", train_x)
        #print("train_y: ", train_y)

        self.model.fit(train_x, train_y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, shuffle=False, verbose=0)
        res = self.model.predict(train_y, batch_size=self.BATCH_SIZE)

        # Get last item only
        last = res[-1].item(0)

        # remove first item (kind of queue)
        # so it contains 1 element.
        # When the next call to the function is done,
        # the history contains two values.
        while len(self.history) > self.BATCH_SIZE+1:
            self.history = np.delete(self.history, 0)

        #print("sending prediction: {}".format(last))

        return last


    def predict(self, value):
        value_reshape = []
        value_reshape = np.append(value_reshape, value)
        value_reshpae = self.reshape_training_set(value_reshape)
        res = self.model.predict(value_reshpae, batch_size=self.BATCH_SIZE)

        # Get last item only
        last = res[-1].item(0)

        return last


    # Reshape the set to fit the samples and batch size
    def reshape_training_set(self, data):
        start_len = len(data)
        entries = len(data)

        # calculate the number of samples
        # each sample contains x TIMESTEPS
        # the number of samples has to be dividable by the batch_size
        samples = int(int(entries / self.TIMESTEPS) / self.BATCH_SIZE) * self.BATCH_SIZE

        # new 1 dimensional length of the list
        new_len = samples * self.TIMESTEPS

        data = data[:new_len]
        data = data.reshape(samples, self.TIMESTEPS, 1)
        # print training_set
        end_len = data.shape[0] * data.shape[1] * data.shape[2]  # samples * batch_size
        percent = (100 * end_len) / start_len
        #print("After reshaping the set to {} samples, {} out of {} entries are used ({}%)".format(samples, end_len, start_len, percent))
        return data




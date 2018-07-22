import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # From 0 to p-t
    for i in range(len(series) - window_size):
        inp = []

        # Pick next window_size elems
        for j in range(i, i + window_size):
            inp.append(series[j])

        # Add input and output
        X.append(inp)
        y.append(series[i + window_size])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model

import string

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    allowed_chars = string.ascii_lowercase + ' ' + '!' + ',' + '.' + ':' + ';' + '?'

    # remove as many non-english characters and character sequences as you can
    for char in text:
        if char not in allowed_chars:
            text = text.replace(char, ' ')

    # shorten any extra dead space created above
    text = text.replace('  ', ' ')

    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    ctr = 0

    # Goes from window_size until the end, and pick previous characters
    for i in range(window_size, len(text), step_size):
        inputs.append(text[ctr:i])
        outputs.append(text[i])
        ctr = ctr + step_size

    return inputs, outputs


# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, 33)))
    model.add(Dense(33, activation='softmax'))
    return model

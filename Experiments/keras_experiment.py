import tensorflow as tf
import numpy as np
from nnutils import *

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import pandas, os

def cumsum(softmax):
    values = tf.split(1, softmax.get_shape()[1], softmax)
    out = []
    prev = tf.zeros_like(values[1])
    for val in values:
        s = prev + val
        out.append(s)
        prev = s
    csum = tf.concat(1, out)
    return csum

from sacred import Experiment
# from sacred.observers import MongoObserver
ex = Experiment('DSB CONVNET EXPERIMENT')
# ex.observers.append(MongoObserver.create())

@ex.config
def config():
    RUN_NAME = 'CRPS-MODEL-3.0'
    DATA_DIR = 'netdata'
    ITERS = 100000
    START_ITER = 0
    MODEL_LOAD_PATH = None
    PIC_WIDTH = 32
    ### Architectural Hyperparameters
    DEPTH_1 = 20         # The output depth of the first convolutional layer
    DEPTH_2 = 40         # The output depth of the second convolutional layer
    DEPTH_3 = 80         # The output depth of the second convolutional layer
    DEPTH_4 = 150        # The output depth of the second convolutional layer
    DEPTH_5 = 150        # The output depth of the second convolutional layer
    DEPTH_6 = 150        # The output depth of the second convolutional layer
    NUM_HIDDEN = 400     # Number of hidden units in the hidden layer
    NUM_OUTPUTS = 600    # Number of output classes in the softmax layer
    KERNEL_X = 3         # The width of the convolution kernel (using same for 1st and 2nd layers)
    KERNEL_Y = 3         # The height of the convolution kernel (using same for 1st and 2nd layers)
    mu = 0.0001
    LEARNING_RATE = 1e-4

    REGULARIZE_BIAS = False


    NUM_INPUTS = 3       # Number of input channels
    NUM_REPS = 64


    TRAIN_LABEL_NOISE_STD = 2.
    TRAIN_LABEL_SMOOTHING_STD = 0.
    DATA_AUGMENTATION = True

@ex.named_config
def augmentation_space_time():
    NUM_INPUTS = 9       # Number of input channels
    NUM_REPS = 144
    TRAIN_LABEL_NOISE_STD = 1.
    TRAIN_LABEL_SMOOTHING_STD = 2.
    DATA_AUGMENTATION = True
    RUN_NAME = 'AUG-EXP'

@ex.capture
def load_data(DATA_DIR):
    train = np.load(os.path.join(DATA_DIR, 'standardized_train.npy'))
    train_labels = np.load(os.path.join(DATA_DIR, 'labels_train.npy'))
    test = np.load(os.path.join(DATA_DIR, 'standardized_test.npy'))
    test_labels = np.load(os.path.join(DATA_DIR, 'labels_test.npy'))


    return train, train_labels, test, test_labels

@ex.automain
def train(DATA_DIR):

    train,train_labels,test,test_labels = load_data()

    lboard = np.load(os.path.join(DATA_DIR, 'standardized_lboard.npy'))
    lboard_labels = np.load(os.path.join(DATA_DIR, 'labels_lboard.npy'))

    train = np.concatenate((train, lboard), axis=1)
    train_labels = np.concatenate((train_labels, lboard_labels), axis=0)
    #print test.shape
    Xv, yv = batch(test, test_labels)
    yv = make_cdf(test_labels)

    model = Sequential()
    # input: 100x100 images with 9 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(9, 32, 32)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #Image down to 16x16
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #Image down to 8x8
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #Image down to 4x4
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #Softmax Output
    model.add(Dense(600))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)




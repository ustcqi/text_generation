# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import numpy as np

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.utils import np_utils

import data_loader


class TextGeneration(object):
    def __init__(self, step=100, nb_epoch=5, batch_size=128,
                 validation_split=0.33, optimizer='adam', loss='categorical_crossentropy'):
        self.step = step
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.loss = loss
        self.optimizer = optimizer
        self.weights_path = '../weights/word-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
        self.X = None
        self.Y = None
        self.dataX = None
        self.dataY = None
        self.nb_vocab = None

    def create_model(self):
        shape_x = np.shape(self.X)
        shape_y = np.shape(self.Y)
        model = Sequential()
        model.add(LSTM(256, input_shape=(shape_x[1], shape_x[2])))
        model.add(Dropout(0.2))
        model.add(Dense(shape_y[1], activation='softmax'))
        return model

    def train(self, save_weights=True):
        model = self.create_model()
        print(model.summary())
        model.compile(loss=self.loss, optimizer=self.optimizer)
        if save_weights:
            checkpoint = ModelCheckpoint(self.weights_path,
                                         monitor='loss',
                                         verbose=1,
                                         save_best_only='True',
                                         mode='min')
            callbacks = [checkpoint]
        else:
            callbacks = []
        model_history = model.fit(self.X, self.Y, nb_epoch=self.nb_epoch,
                                  batch_size=self.batch_size, callbacks=callbacks)
        print(model_history.history.keys())

        self.generate_text(model)

        return model_history

    def load_model(self, weights_filename):
        model = self.create_model()
        model.load_weights(weights_filename)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def generate_text(self, model):
        start = np.random.randint(0, len(self.X)-1)
        pattern = self.dataX[start]
        result = []
        print(len(pattern))
        print('Seed:')
        print(' '.join([self.idx2word[idx] for idx in pattern]))
        for i in xrange(200):
            x = np.reshape(pattern, (1, len(pattern), 1)) / float(self.nb_vocab)
            prediction = model.predict(x)
            idx = np.argmax(prediction)
            result.append(self.idx2word[idx])
            pattern.append(idx)
            pattern = pattern[1:len(pattern)]
        print('\nGenerated Sequence:')
        print(' '.join(result))



def main():
    data_root = os.getcwd()[:-3] + 'data/'
    filename = data_root + "wonderland.txt"
    print(filename)
    step = 100

    dataX, dataY = data_loader.get_train_data(filename, step=step)
    nb_vocab = data_loader.get_nb_vocab(filename)
    idx2word = data_loader.get_idx2word(filename)
    X, Y = data_loader.transform(dataX, dataY, nb_vocab=nb_vocab)
    print(np.shape(X), np.shape(Y))
    text_generation = TextGeneration(nb_epoch=5, step=step)
    text_generation.X = X
    text_generation.Y = Y
    text_generation.dataX = dataX
    text_generation.dataY = dataY
    text_generation.nb_vocab = nb_vocab
    text_generation.idx2word = idx2word

    model_history = text_generation.train()
    """
    weights_filename = '../weights/word-weights-improvement-04-6.0882.hdf5'
    model = text_generation.load_model(X, Y, weights_filename)
    print(model)
    idx2word = get_idx2word(filename)
    text_generation.generate_text(model, dataX, idx2word, nb_vocab)
    """


main()





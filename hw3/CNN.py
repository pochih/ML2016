# coding=utf-8
from __future__ import print_function
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Nadam
import cPickle as pickle
import json
import time
import parser as ps
import numpy as np
import sys

nb_classes = 10
batch_size = 100
nb_epoch = 350
validPercent = 15
LOAD_FLAG = False
LOAD_MODEL_FILE = "trained_model"
MODEL_FILE = sys.argv[2]
dataPath = sys.argv[1]

# slow data
# labels[0-9][0-499][0-3071]
ts = time.time()
print("Loading labeled data......")
labels = pickle.load(open(dataPath + 'all_label.p', "rb"))
X_train, Y_train = ps.parseTrain(labels, nb_classes, 'rgb')
# print('labels[9][499]', labels[9][499][1023], labels[9][499][2047], labels[9][499][3071])
# print('X_train[4999][31][31]', X_train[4999][31][31])
te = time.time()
print(te-ts, 'secs')

# unlabels[0-44999][0-3071]
# ts = time.time()
# print("Loading unlabeled data......")
# unlabels = pickle.load(open(dataPath + 'all_unlabel.p', "rb"))
# X_unlabel, Y_unlabel = ps.parseUnlabel(unlabels, nb_classes, 'rgb')
# # print('unlabels[44999]', unlabels[44999][1023], unlabels[44999][2047], unlabels[44999][3071])
# # print('X_unlabel[44999][31][31]', X_unlabel[44999][31][31])
# te = time.time()
# print(te-ts, 'secs')

# tests['ID'][0-9999], tests['data'][0-9999], tests['labels'][0-9999]
ts = time.time()
print("Loading test data......")
tests = pickle.load(open(dataPath + 'test.p', "rb"))
X_test, Y_test = ps.parseTest(tests, nb_classes, 'rgb')
# print('tests["data"][0]', tests["data"][0][0], tests["data"][0][1024], tests["data"][0][2048])
# print('X_test[0][0][0]', X_test[0][0][0])
te = time.time()
print(te-ts, 'secs')

# pickle.dump((X_train, Y_train), open("fast_all_label", "wb"), True)
# pickle.dump((X_unlabel, Y_unlabel), open("fast_all_unlabel", "wb"), True)
# pickle.dump((X_test, Y_test), open("fast_test", "wb"), True)

## fast data
# ts = time.time()
# (X_train, Y_train) = pickle.load(open("fast_all_label", "rb"))
# # (X_unlabel, Y_unlabel) = pickle.load(open("fast_all_unlabel", "rb"))
# (X_test, Y_test) = pickle.load(open("fast_test", "rb"))
# te = time.time()
# print('Loading data......', te-ts, 'secs')
# '''(5000, 32, 32, 3) (5000, 10) (10000, 32, 32, 3) (10000, 10) (45000, 32, 32, 3) (45000, 10)'''
# print('shape: X_train', X_train.shape, 'Y_train', Y_train.shape, 'X_test', X_test.shape, 'Y_test', Y_test.shape, 'X_unlabel', X_unlabel.shape, 'Y_unlabel', Y_unlabel.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# add model
model = Sequential()
if LOAD_FLAG:
    model = load_model(LOAD_MODEL_FILE)
else:
    model.add(Convolution2D(32, 4, 4, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 4, 4))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.55))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.55))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.55))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # start CNN
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

X_validation, Y_validation = ps.parseValidation(X_train, Y_train, nb_classes, len(X_train)*validPercent/100, _type='rgb')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_validation, Y_validation), shuffle=True)

# save model
print("Saving model......")
model.save(MODEL_FILE)

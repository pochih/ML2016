# coding=utf-8
from __future__ import print_function
from keras.models import Sequential
from keras.models import load_model
import cPickle as pickle
import time
import parser as ps
import numpy as np
import sys

nb_classes = 10
out = open(sys.argv[3], 'w')
LOAD_MODEL_FILE = sys.argv[2]
dataPath = sys.argv[1]

# slow data
# labels[0-9][0-499][0-3071]
# ts = time.time()
# print("Loading labeled data......")
# labels = pickle.load(open(dataPath + 'all_label.p', "rb"))
# X_train, Y_train = ps.parseTrain(labels, nb_classes, 'rgb')
# # print('labels[9][499]', labels[9][499][1023], labels[9][499][2047], labels[9][499][3071])
# # print('X_train[4999][31][31]', X_train[4999][31][31])
# te = time.time()
# print(te-ts, 'secs')

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

model = load_model(LOAD_MODEL_FILE)
print(model.summary())

X_test = X_test.astype('float32') / 255

# predict
print("Predicting test data......")
result = model.predict(X_test)
out.write('ID,class\n')
for i in range(len(result)):
    out.write(str(i) + ',' + str(np.argmax(result[i])) + '\n')
print('result[9999]', result[9999])
print('result[9996]', result[9996])

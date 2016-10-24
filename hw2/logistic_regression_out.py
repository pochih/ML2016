# coding=utf-8

import sys
import cPickle as pickle
import numpy as np

ORDER = 1
SCALING = False  # feature scaling

def mapTestOrder(testData, ORDER):
    # 生成 test data
    per_order_size = len(testData[0]) - 1
    X_t = np.ones(shape=(len(testData), per_order_size * ORDER + 1))    

    for i in range(len(testData)):
        for j in range(ORDER):
            X_t[i][j*per_order_size+1:(j+1)*per_order_size+1] = [(x ** (j+1)) for x in testData[i][1:]]
    return X_t

def predict(theta, X_t):
    print "Shape", X_t.shape, theta.shape
    y_t = X_t.dot(theta)
    return [1 if y >= 0.5 else 0 for y in y_t]

if __name__ == '__main__':
    theta = pickle.load(open(sys.argv[1], "rb"))

    # 處理 raw test data
    rawTestData = open(sys.argv[2], 'r').read().split('\n')[:-1]
    for i in range(len(rawTestData)):
        rawTestData[i] = rawTestData[i].split(',')
        for j in range(len(rawTestData[i])):
            rawTestData[i][j] = float(rawTestData[i][j])

    # 生成 testing data
    X_t = mapTestOrder(rawTestData, ORDER)

    # feature scaling
    if SCALING:
        X_t = (X_t - mean(X, axis=0)) / (std(X, axis=0)).clip(min=0.000001)

    # predict
    y_t = predict(theta, X_t)

    # output
    out = open(sys.argv[3], 'w')
    out.write('id,label\n')
    for i in range(len(y_t)):
        out.write(str(i+1) + ',' + str(y_t[i]) + '\n')
    out.close()
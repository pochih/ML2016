# coding=utf-8

import sys
import network
import cPickle as pickle
import numpy as np
import random

ORDER = 1
splitPercent = 15

def mapTrainOrder(data, ORDER):

    # 生成 train data
    per_order_size = len(data[0]) - 2
    X = []

    for i in range(len(data)):
        X_tmp = np.zeros(shape=(per_order_size * ORDER, 1))
        for j in range(ORDER):
            X_tmp[j*per_order_size:(j+1)*per_order_size, 0] = [(x ** (j+1)) for x in data[i][1:-1]]
        y_tmp = np.zeros(shape=(2, 1))
        y_tmp[int(data[i][-1])] = 1
        X.append((X_tmp, y_tmp))
    print "trainData[0][0].shape:{0}, trainData[0][1].shape:{1}".format(X[0][0].shape, X[0][1].shape)
    return X

def mapTestOrder(data, ORDER):

    # 生成 test data
    per_order_size = len(data[0]) - 2
    X = []

    for i in range(len(data)):
        X_tmp = np.zeros(shape=(per_order_size * ORDER, 1))
        for j in range(ORDER):
            X_tmp[j*per_order_size:(j+1)*per_order_size, 0] = [(x ** (j+1)) for x in data[i][1:-1]]
        y_tmp = int(data[i][-1])
        X.append((X_tmp, y_tmp))
    print "testData[0][0].shape:{0}".format(X[0][0].shape)
    return X

if __name__ == '__main__':

    # 處理 raw train data
    rawTrainData = open(sys.argv[1], 'r').read().split('\r\n')[:-1]
    for i in range(len(rawTrainData)):
        rawTrainData[i] = rawTrainData[i].split(',')
        for j in range(len(rawTrainData[i])):
            rawTrainData[i][j] = float(rawTrainData[i][j])

    # 洗牌
    random.shuffle(rawTrainData)

    # 依照比例 (splitPercent) 生成 train data
    trainData = mapTrainOrder(rawTrainData[len(rawTrainData)*splitPercent/100:], ORDER)

    # 依照比例 (splitPercent) 生成 test data
    testData = mapTestOrder(rawTrainData[:len(rawTrainData)*splitPercent/100], ORDER)

    # 建立 neural network
    net = network.Network([57, 35, 15, 10, 5, 2])

    # SGD(train data, iterations, batch-size, learning rate, test data)
    W, B = net.SGD(trainData, 30000, 10, 0.015, test_data=testData)

    # 儲存 model
    pickle.dump((W, B), open(sys.argv[2], "wb"), True)

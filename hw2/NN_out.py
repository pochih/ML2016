# coding=utf-8

import sys
import cPickle as pickle
import numpy as np

ORDER = 1

def mapTestOrder(testData, ORDER):

    # 生成 test data
    per_order_size = len(testData[0]) - 1
    X_t = []

    for i in range(len(testData)):
    	X_t_tmp = np.zeros(shape=(per_order_size * ORDER, 1))    
        for j in range(ORDER):
            X_t_tmp[j*per_order_size:(j+1)*per_order_size, 0] = [(x ** (j+1)) for x in testData[i][1:]]
        X_t.append(X_t_tmp)
    return X_t

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def feedForward(X, weights, biases):

    # forwarding	
    for w, b in zip(weights, biases):
        X = sigmoid(np.dot(w, X)+b)
    return X

if __name__ == '__main__':
    weights, biases = pickle.load(open(sys.argv[1], "rb"))

    # 處理 raw test data
    rawTestData = open(sys.argv[2], 'r').read().split('\n')[:-1]
    for i in range(len(rawTestData)):
        rawTestData[i] = rawTestData[i].split(',')
        for j in range(len(rawTestData[i])):
            rawTestData[i][j] = float(rawTestData[i][j])

    # 生成 test data
    testData = mapTestOrder(rawTestData, ORDER)

    # 檢驗結果
    testResults = [np.argmax(feedForward(x, weights, biases)) for x in testData]

    # output
    out = open(sys.argv[3], 'w')
    out.write('id,label\n')
    for i in range(len(testResults)):
    	out.write(str(i+1) + ',' + str(testResults[i]) + '\n')

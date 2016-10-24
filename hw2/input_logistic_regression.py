# coding=utf-8

from numpy import zeros, ones, e, log, random, mean, std, clip
import sys
import time
import cPickle as pickle

ORDER = 1                   # order
LAMBDA = 11                 # regularization rate
SCALING = False             # feature scaling
ALPHA = 0.00002             # learning rate
MAX_ITERATION = 1000000     # max iterations
TIME_MAX = float("inf")     # time max (10 mins)
output = open('submit_50min', 'w')
thetaOut = 'theta_50min'
thetaData = open('theta_40min', 'r').read().split(',')
thetaData = thetaData[:-1]
for i in range(len(thetaData)):
    thetaData[i] = float(thetaData[i])

def sigmoid(X):

    den = 1.0 + e ** (-1.0 * X)
    d = 1.0 / den

    return d

def countError(theta, X, y):

    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    thetaR = theta[1:, 0]

    # count error
    E = (1.0 / m) * ((-y.T.dot(log(h + 0.00000000001))) - ((1 - y.T).dot(log(1.00000000001 - h)))) + (LAMBDA / (2.0 * m)) * (thetaR.T.dot(thetaR))

    return E

def countGradient(theta, X, y):

    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    thetaR = theta[1:, 0]

    # gradient of constant
    delta = h - y
    sumDelta = delta.T.dot(X[:, 1])
    grad1 = (1.0 / m) * sumDelta

    # gradient of non-constant
    XR = X[:, 1:X.shape[1]]
    sumDeltaR = delta.T.dot(XR)
    grad = (1.0 / m) * (sumDeltaR + LAMBDA * (thetaR.T))

    # print delta.shape, XR.shape, sumDeltaR.shape, thetaR.shape, grad.shape

    out = zeros(shape=(grad.shape[0], grad.shape[1]+1))

    out[:, 0] = grad1
    out[:, 1:] = grad

    return out.T

def logistic_regression(X, y):

    # initial theta
    # theta = random.uniform(-.01, .01, X.shape[1])
    theta = zeros(shape=(X.shape[1], 1))
    for i in range(X.shape[1]):
        theta[i] = thetaData[i]

    # Start Iterations
    iter = 0
    oldError = 10000000000
    error = -1
    alpha = ALPHA
    alphaFlag = 0
    tstart = time.time()
    tend = time.time()
    while abs(tend - tstart) < TIME_MAX:
        oldError = error

        # count error, gradient
        grad = countGradient(theta, X, y)

        # update theta
        theta = theta - alpha * grad

        iter += 1
        if iter >= MAX_ITERATION:
            print "reach max_iter"
            break

        if iter % 5 == 0:
            error = countError(theta, X, y)
            print("Iteration %d | Error: %f | alpha: %.10f" % (iter, error, alpha))
            print("Time: %f" % (tend - tstart))
            if error < oldError:
                if ((oldError - error) / oldError) < 0.01:
                    alphaFlag += 1
            if error > oldError:
                alpha = alpha / 2
            if alphaFlag >= 10:
                alpha *= 1.1
                alphaFlag = 0

        tend = time.time()

    return theta

def mapTrainOrder(trainData, ORDER):
    # 生成 train data
    per_order_size = len(trainData[0]) - 2
    X = ones(shape=(len(trainData), per_order_size * ORDER + 1))
    y = zeros((len(trainData), 1))

    for i in range(len(trainData)):
        for j in range(ORDER):
            X[i][j*per_order_size+1:(j+1)*per_order_size+1] = [(x ** (j+1)) for x in trainData[i][1:-1]]
        y[i] = trainData[i][-1:]
    print("Maping: X.shape: (%d, %d), y.shape(%d, %d)" % (X.shape[0], X.shape[1], y.shape[0], y.shape[1]))
    return X, y

def mapTestOrder(testData, ORDER):
    # 生成 test data
    per_order_size = len(testData[0]) - 1
    X_t = ones(shape=(len(testData), per_order_size * ORDER + 1))    

    for i in range(len(testData)):
        for j in range(ORDER):
            X_t[i][j*per_order_size+1:(j+1)*per_order_size+1] = [(x ** (j+1)) for x in testData[i][1:]]
    return X_t

def predict(theta, X_t):
    print "Shape", X_t.shape, theta.shape
    y_t = X_t.dot(theta)
    return [1 if y >= 0.5 else 0 for y in y_t]

if __name__ == '__main__':
    # 處理 raw train data
    rawTrainData = open('spam_data/spam_train.csv', 'r').read().split('\r\n')[:-1]
    for i in range(len(rawTrainData)):
        rawTrainData[i] = rawTrainData[i].split(',')
        for j in range(len(rawTrainData[i])):
            rawTrainData[i][j] = float(rawTrainData[i][j])

    # 處理 raw test data
    rawTestData = open('spam_data/spam_test.csv', 'r').read().split('\n')[:-1]
    for i in range(len(rawTestData)):
        rawTestData[i] = rawTestData[i].split(',')
        for j in range(len(rawTestData[i])):
            rawTestData[i][j] = float(rawTestData[i][j])

    X, y = mapTrainOrder(rawTrainData, ORDER)

    # feature scaling
    if SCALING:
        X = (X - mean(X, axis=0)) / (std(X, axis=0)).clip(min=0.000001)

    # logistic regression
    theta = logistic_regression(X, y)
    print theta

    X_t = mapTestOrder(rawTestData, ORDER)

    # predict
    y_t = predict(theta, X_t)

    # 儲存 model
    pickle.dump(theta, open(thetaOut, "wb"), True)

    # output
    output.write('id,label\n')
    for i in range(len(y_t)):
        output.write(str(i+1) + ',' + str(y_t[i]) + '\n')
    output.close()

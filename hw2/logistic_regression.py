# coding=utf-8

from numpy import zeros, ones, e, log, random, mean, std, clip, array
import sys
import time
import cPickle as pickle

ORDER = 1                   # order
LAMBDA = 10                 # regularization rate
SCALING = False             # feature scaling
ALPHA = 0.001               # learning rate
MAX_ITERATION = 1000000     # max iterations
TIME_MAX = 60*10-3          # time max (10 mins)

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
    theta = random.uniform(-.01, .01, X.shape[1])
    theta.shape = (X.shape[1], 1)

    # Start Iterations
    iter = 0
    oldError = 10000000000
    error = -1
    alpha = ALPHA
    alphaFlag = 0
    tstart = time.time()
    tend = time.time()
    while abs(tend - tstart) < TIME_MAX:

        # count error, gradient
        grad = countGradient(theta, X, y)

        if iter % 10 == 0:
            oldError = error
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

        # update theta
        theta = theta - alpha * grad

        iter += 1
        if iter >= MAX_ITERATION:
            print "reach max_iter"
            break

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

if __name__ == '__main__':
    # 處理 raw train data
    rawTrainData = open(sys.argv[1], 'r').read().split('\r\n')[:-1]
    for i in range(len(rawTrainData)):
        rawTrainData[i] = rawTrainData[i].split(',')
        for j in range(len(rawTrainData[i])):
            rawTrainData[i][j] = float(rawTrainData[i][j])

    # 生成 training data
    X, y = mapTrainOrder(rawTrainData, ORDER)

    # feature scaling
    if SCALING:
        X = (X - mean(X, axis=0)) / (std(X, axis=0)).clip(min=0.000001)

    # logistic regression
    theta = logistic_regression(X, y)
    print theta

    # 儲存 model
    pickle.dump(theta, open(sys.argv[2], "wb"), True)

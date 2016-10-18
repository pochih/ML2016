# coding=utf-8

from numpy import loadtxt, where, zeros, e, array, log, ones, mean, where, random
import sys
import time

LAMBDA = 1 					# regularization rate
ALPHA = 0.001				# learning rate
MAX_ITERATION = 10000000 	# max iterations
TIME_MAX = 10 * 60 - 3 		# time max (10 mins)

def sigmoid(X):

    den = 1.0 + e ** (-1.0 * X)
    d = 1.0 / den

    return d

def countError(theta, X, y):

    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    thetaR = theta[1:, 0]

    # count error
    E = (1.0 / m) * ((-y.T.dot(log(h))) - ((1 - y.T).dot(log(1.00000000001 - h)))) + (LAMBDA / (2.0 * m)) * (thetaR.T.dot(thetaR))

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
        oldError = error

        # count error, gradient
        grad = countGradient(theta, X, y)

        # update theta
        theta = theta - alpha * grad

        iter += 1
        if iter >= MAX_ITERATION:
            print "reach max_iter"
            break

        if iter % 1000 == 0:
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

def predict(theta, X_t):
	print theta.shape, X_t.shape
	y_t = X_t.dot(theta[1:, :])
	y_t += theta[0]
	return y_t

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

	# 生成 train data
    X = ones(shape=(len(rawTrainData), len(rawTrainData[0])-1))
    y = zeros((len(rawTrainData), 1))

    for i in range(len(rawTrainData)):
        X[i][1:] = rawTrainData[i][1:-1]
        y[i] = rawTrainData[i][-1:]

    # logistic regression
    theta = logistic_regression(X, y)
    print theta

    # 生成 test data
    X_t = ones(shape=(len(rawTestData), len(rawTestData[0])-1))    
    for i in range(len(rawTestData)):
        X_t[i] = rawTestData[i][1:]

    # predict
    y_t = predict(theta, X_t)

    # output
    thetaOut = open('theta', 'w')
    for i in range(len(theta)):
        thetaOut.write(str(theta[i, 0])+',')
    thetaOut.close()
    output = open('submit', 'w')
    output.write('id,label\n')
    for i in range(len(y_t)):
        if y_t[i] >= 0.5:
            output.write(str(i+1) + ',' + str(1) + '\n')
        else:
            output.write(str(i+1) + ',' + str(0) + '\n')
    output.close()
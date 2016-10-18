# coding=utf-8

from numpy import loadtxt, where, zeros, e, array, log, ones, mean, where, random
import sys
import time

LAMBDA = 1 					# regularization rate
ALPHA = 0.0001				# learning rate
MAX_ITERATION = 10000000 	# max iterations
TIME_MAX = float("inf")     # time max (10 mins)
output = open('submit_1000W', 'w')
thetaOut = open('theta_1000W', 'w')
thetaData = open('theta_100W', 'r').read().split(',')
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

    #output
    for i in range(len(theta)):
        thetaOut.write(str(theta[i, 0])+',')
    thetaOut.close()
    output.write('id,label\n')
    for i in range(len(y_t)):
        if y_t[i] >= 0.5:
            output.write(str(i+1) + ',' + str(1) + '\n')
        else:
            output.write(str(i+1) + ',' + str(0) + '\n')




















 	# theta_ = -0.08021634,-0.06407701,-0.22637217,-0.09220102, 0.12697138, 0.23258177, 0.14626913, 0.45762226, 0.24494365, 0.11197205, 0.03437675, 0.07892411,-0.57600145, 0.01820763,-0.04041104, 0.14220946, 0.52035001, 0.26751088, 0.17918759,-0.07263395, 0.27570445, 0.20967468, 0.14043835, 0.49453646, 0.3308254 ,-1.76290734,-0.77672023,-0.86422397,-0.23233146,-0.2666614 ,-0.25353461,-0.176225  ,-0.08283169,-0.5680406 ,-0.07167912,-0.3220465 ,-0.18961279,-0.55614776,-0.07796594,-0.30402326,-0.11635928,-0.22826372,-0.55926946,-0.17871724,-0.39081199,-0.68596169,-0.85555956,-0.04991995,-0.20781785,-0.28912937,-0.41249792,-0.08133073, 0.33882218, 0.34022589, 0.07850789,-0.03392957, 0.02420506,-0.0073015,


	# 只用長度判斷
	# yes_sum = 0
	# no_sum = 0
	# yes_num = 0
	# no_num = 0
	# yes_max = -1
	# yes_min = float('inf')
	# no_max = -1
	# no_min = float('inf')
	# for i in range(len(rawTrainData)):
	# 	if rawTrainData[i][58] == 1:
	# 		yes_sum += rawTrainData[i][57]
	# 		yes_num += 1
	# 		if rawTrainData[i][57] > yes_max:
	# 			yes_max = rawTrainData[i][57]
	# 		if rawTrainData[i][57] < yes_min:
	# 			yes_min = rawTrainData[i][57]
	# 	elif rawTrainData[i][58] == 0:
	# 		no_sum += rawTrainData[i][57]
	# 		no_num += 1
	# 		if rawTrainData[i][57] > no_max:
	# 			no_max = rawTrainData[i][57]
	# 		if rawTrainData[i][57] < no_min:
	# 			no_min = rawTrainData[i][57]
	# print yes_sum/yes_num, no_sum/no_num, yes_max, yes_min, no_max, no_min, yes_num, no_num

	# output.write('id,label\n')

	# for i in range(len(rawTestData)):
	# 	length = rawTrainData[i][57]
	# 	if abs(length - 467.91956242) >= abs(length - 161.046178995):
	# 		output.write(str(i+1) + ',' + str(0) + '\n')
	# 	else:
	# 		if np.random.random(1)[0] > 0.5:
	# 			output.write(str(i+1) + ',' + str(1) + '\n')
	# 		else:
	# 			output.write(str(i+1) + ',' + str(0) + '\n')
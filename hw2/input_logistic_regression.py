# coding=utf-8

from numpy import zeros, ones, e, log, random, mean, std, clip
import sys
import time

ORDER = 1                   # order
LAMBDA = 11 				# regularization rate
SCALING = False             # feature scaling
ALPHA = 0.00002			    # learning rate
MAX_ITERATION = 1000000 	# max iterations
TIME_MAX = float("inf")     # time max (10 mins)
output = open('submit_50min', 'w')
thetaOut = open('theta_50min', 'w')
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
    y_t = X_t.dot(theta)
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
    # X = ones(shape=(len(rawTrainData), len(rawTrainData[0])-1))
    # y = zeros((len(rawTrainData), 1))

    # for i in range(len(rawTrainData)):
    #     X[i][1:] = rawTrainData[i][1:-1]
    #     y[i] = rawTrainData[i][-1:]

    X, y = mapTrainOrder(rawTrainData, ORDER)

    # feature scaling
    if SCALING:
        X = (X - mean(X, axis=0)) / (std(X, axis=0)).clip(min=0.000001)

    # logistic regression
    theta = logistic_regression(X, y)
    print theta
    # theta_ = [2.13432964e+00, 2.34070528e+00,  6.73792993e-01,  7.50533528e+00,  6.03945603e+00,  2.79855535e+01,  8.99556902e+00,  2.49554361e+01,  1.62458935e+01,  3.66540964e+00,  2.78910646e+00,  5.33586432e+00, -2.99748994e+01,  4.11097522e+00,  1.23080238e+00,  5.45134085e+00,  4.91808205e+01,  1.41445727e+01,  1.65458203e+01,  7.92044403e+01,  1.65722375e+01,  6.16839592e+01,  9.18011360e+00,  2.24898013e+01,  2.01670395e+01, -1.45632269e+02, -7.04871774e+01, -4.33379690e+01, -3.04610679e+01, -1.71391693e+01, -2.35310045e+01, -1.43239808e+01, -1.02063832e+01, -2.44447428e+01, -1.00829603e+01, -2.57323205e+01, -1.80207841e+01, -3.53347430e+01, -1.03332260e+00, -2.05250070e+01, -9.14643095e+00, -1.04193341e+01, -1.93846935e+01, -1.31065380e+01, -1.28645297e+01, -2.64203913e+01, -4.03208910e+01, -1.36112703e+00, -7.39105063e+00, -8.93810280e+00, -1.59068629e+01, -3.01339751e+00,  4.04700809e+01,  1.41828631e+01,  2.92992443e+00, -8.52413445e+01, -4.27999810e+02, -9.56809000e+03,  1.79762152e+00, -2.76028908e+01,  8.75956478e+00,  1.53406519e+02,  6.08203441e+01,  9.91115992e+00,  4.26644644e+01,  4.85635072e+01,  4.58478548e+00, -4.95366244e+00,  9.80722760e-01, -7.35992495e+01,  1.10471840e+01,  1.78463010e+00,  8.59686803e+00,  1.73067353e+02,  2.63027524e+01,  5.44672362e+01,  3.62469180e+02,  8.87019362e+01,  1.53804037e+02,  9.40332098e+01,  3.57692947e+01,  5.81890364e+01, -5.90396617e+02, -1.62662919e+02, -9.82389187e+01, -4.71704220e+01, -2.82717487e+01, -3.38719786e+01, -1.72561637e+01, -1.33807883e+01, -6.18018606e+01, -1.32429461e+01, -4.17771092e+01, -1.98462257e+01, -3.94464114e+01, -1.90939773e+00, -4.71227281e+01, -1.58678515e+01, -2.37575948e+01, -4.52474708e+01, -1.29783375e+01, -2.88018447e+01, -5.67225074e+01, -2.15404293e+02, -1.24679845e+00, -9.11676882e+00, -2.21966958e+01, -8.93801302e+00, -7.65662061e-01,  2.28083315e+01,  1.67907722e+01,  2.03025957e+01, -1.04009679e+03,  3.21917895e+02,  1.99126567e+02]
    # theta = zeros(shape=(115,1))
    # for i in range(115):
    #     theta[i] = theta_[i]

    # 生成 test data
    # X_t = ones(shape=(len(rawTestData), len(rawTestData[0])-1))    
    # for i in range(len(rawTestData)):
    #     X_t[i] = rawTestData[i][1:]

    X_t = mapTestOrder(rawTestData, ORDER)

    # predict
    y_t = predict(theta, X_t)

    # output
    for i in range(len(theta)):
        thetaOut.write(str(theta[i, 0])+',')
    thetaOut.close()
    output.write('id,label\n')
    for i in range(len(y_t)):
        if y_t[i] >= 0.5:
            output.write(str(i+1) + ',' + str(1) + '\n')
        else:
            output.write(str(i+1) + ',' + str(0) + '\n')
    output.close()



















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
# coding=utf-8
import numpy as np

rawTrainData = open('train.csv', 'r').read().split('\r\n')
rawTestData = open('test_X.csv', 'r').read().split('\n')
output = open('kaggle_best.csv', 'w')
thetaOut = open('theta_kaggle_best.csv', 'w')
thetaData = open('theta_small_linux9_2', 'r').read().split(',')
thetaData = thetaData[:-1]

# 存 features
features = []
# train data，把每一天的同一種 feature 串在一起
trainData = []
# 連續 9 小時的 data + 第 10 小時的 PM2.5
contTrainData = []
# test data
testData = []

## gradient descent ##
alpha = 0.000000005 	    # learning rate
ep = 0.0001 	            # convergence criteria
lamda = 3.01 	            # regularization rate
MAX_ITERATION = 150000

def countCost(contTrainData, theta, lamda):
	# total error, C(theta) = 1/2*sum((theta dot X - Y)^2) + lamda*sum(W^2)
    C = 0
    Loss = []
    for i in range(len(contTrainData)):
    	tmp = theta[0]
        # 18*9 個 features 的 1 次方
    	for j in range(18*9):
            tmp += (theta[j+1] * contTrainData[i][0][j])
        # # 最後一個小時 18 個 features 的 2 次方
        # for k in range(18):
        #     tmp += (theta[k+163] * (contTrainData[i][0][k*9+8] ** 2))
    	tmp -= contTrainData[i][1]
    	Loss.append(tmp)
    	C += (tmp ** 2)
    	# print 'Round'+str(i), tmp, C
    C = C * 0.5
    for i in range(len(theta)):
    	C += (lamda * (theta[i] ** 2))
    	# print i, C
    return C, Loss

def gradient_descent(alpha, lamda, contTrainData, ep=0.0001, max_iter=10000):
    iter = 0
    num = len(contTrainData) # number of samples

    # initial theta
    # theta = np.random.random(18*9+1)
    # theta = np.ones(18*9+1)
    # theta = np.random.uniform(-.01, .01, (18*9+1))
    # theta = np.zeros(18*9+1)
    for i in range(len(thetaData)):
        thetaData[i] = float(thetaData[i])
    theta = thetaData
    # theta = np.random.uniform(-.01, .01, (18*9+1))

    # Iterate Loop
    iter = 0
    oldCost = float("inf")
    cost = -1
    alphaFlag = 0
    while oldCost != cost:
        oldCost = cost

        # total error, C(theta) = 1/2*sum((theta dot X - Y)^2) + lamda*sum(W^2)
    	cost, loss = countCost(contTrainData, theta, lamda)
        print("Iteration %d | Cost: %f | alpha: %.15f" % (iter, cost, alpha))

        # count gradient
    	gradient = np.zeros(18*9+1)
        for i in range(len(theta)):
        	gradient[i] += 2 * lamda * theta[i]
        gradient[0] = sum(loss)
        for i in range(len(contTrainData)):
    		for k in range(18*9):
    			gradient[k+1] += loss[i] * contTrainData[i][0][k]

        # update theta
        if cost < oldCost:
        	if ((oldCost - cost) / oldCost) < 0.01:
        		alphaFlag += 1
        if cost > oldCost:
        	alpha = alpha / 2
        if alphaFlag >= 10:
        	alpha *= 1.1
        	alphaFlag = 0
        for i in range(len(theta)):
        	theta[i] -= alpha * gradient[i]

        iter += 1
        if iter >= max_iter:
            print "reach max_iter"
            break
    return theta


if __name__ == '__main__':

    ### 處理 train data ###
    
    # 先把每一行 data 用 ',' 分開
    for i in range(0, len(rawTrainData)):
    	rawTrainData[i] = rawTrainData[i].split(',')
    
    # 建 feature
    for i in range(1, 19):
    	feature = rawTrainData[i][2]
    	features.append(feature)
    
    # 生成 trainData
    for i in range(18):
    	trainData.append([])
    for i in range(240):
    	for j in range(18):
    		line = i * 18 + j + 1
    		trainData[j] += rawTrainData[line][3:]
    
    # string 轉 float
    for i in range(len(trainData)):
    	for j in range(len(trainData[i])):
    		if trainData[i][j] == 'NR':
    			trainData[i][j] = 0
    		else:
    			trainData[i][j] = float(trainData[i][j])
    
    # 生成 contTrainData (從 trainData 中抽取連續 9 小時的 data + 第 10 小時的 PM2.5)
    for i in range(12):
    	for j in range(471):
    		location = i * 480 + j
    		tmpList = []
    		for k in range(18):
    			tmpList += trainData[k][location:location+9]
    		contTrainData.append((tmpList, trainData[9][location+9]))
    
    
    # gredient decent
    theta = gradient_descent(alpha, lamda, contTrainData, ep, max_iter=MAX_ITERATION)
    print theta
    for i in range(len(theta)):
    	thetaOut.write(str(theta[i])+',')
    thetaOut.close()
    
    
    ### 處理 test data ###
    
    # 去除末端空行
    rawTestData = rawTestData[0:-1]
    
    # 先把每一行 data 用 ',' 分開
    for i in range(0, len(rawTestData)):
    	rawTestData[i] = rawTestData[i].split(',')
    
    # 生成 testData
    for i in range(240):
    	tmpList = []
    	for j in range(18):
    		line = i * 18 + j
    		tmpList += rawTestData[line][2:]
    	testData.append(tmpList)
    
    # string 轉 float
    for i in range(len(testData)):
    	for j in range(len(testData[i])):
    		if testData[i][j] == 'NR':
    			testData[i][j] = 0
    		else:
    			testData[i][j] = float(testData[i][j])
    
    # 用 theta 計算出值，然後 output
    output.write('id,value\n')
    for i in range(len(testData)):
        sum = theta[0]
        for j in range(18*9):
            sum += (testData[i][j] * theta[j+1])
        output.write('id_' + str(i) + ',' + str(sum) + '\n')
    output.close()

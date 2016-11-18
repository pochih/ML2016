import numpy as np
import random

# [0-9][0-499][0-3071] to (5000, 32, 32, 3), (5000, 10)
def parseTrain(data, nb_classes, _type='b/w'):
	pixels = int(len(data[0][0]) ** 0.5)
	channels = 1
	if _type == 'rgb':
		pixels = int((len(data[0][0]) / 3) ** 0.5)
		channels = 3
	classes = nb_classes
	data_len = sum([len(x) for x in data])
	X = np.zeros((data_len, pixels, pixels, channels))
	Y = np.zeros((data_len, classes))
	index = 0
	for i in range(len(data)):
		class_len = len(data[i])
		Y[index:index+class_len, i] = 1
		for j in range(0, class_len):
			# tmp = np.array([[data[i][j][x],data[i][j][x+pixels*pixels],data[i][j][x+pixels*pixels*2]] for x in range(pixels*pixels)])
			for k in range(pixels):
				for l in range(pixels):
					index2 = k*pixels + l
					X[index+j][k][l] = [data[i][j][index2], data[i][j][index2+pixels*pixels], data[i][j][index2+pixels*pixels*2]]
		index += class_len

	return X, Y

def parseUnlabel(data, nb_classes, _type='b/w'):
	pixels = int(len(data[0]) ** 0.5)
	channels = 1
	if _type == 'rgb':
		pixels = int((len(data[0]) / 3) ** 0.5)
		channels = 3
	classes = nb_classes
	data_len = len(data)
	X = np.zeros((data_len, pixels, pixels, channels))
	Y = np.zeros((data_len, classes))
	for i in range(data_len):
		for k in range(pixels):
			for l in range(pixels):
				index = k*pixels + l
				X[i][k][l] = [data[i][index], data[i][index+pixels*pixels], data[i][index+pixels*pixels*2]]

	return X, Y

def parseTest(data, nb_classes, _type='b/w'):
	pixels = int(len(data['data'][0]) ** 0.5)
	channels = 1
	if _type == 'rgb':
		pixels = int((len(data['data'][0]) / 3) ** 0.5)
		channels = 3
	classes = nb_classes
	data_len = len(data['data'])
	X = np.zeros((data_len, pixels, pixels, channels))
	Y = np.zeros((data_len, classes))
	for i in range(data_len):
		for k in range(pixels):
			for l in range(pixels):
				index = k*pixels + l
				X[i][k][l] = [data['data'][i][index], data['data'][i][index+pixels*pixels], data['data'][i][index+pixels*pixels*2]]

	return X, Y

def parseValidation(X_train, Y_train, nb_classes, size, _type='b/w'):
	pixels = X_train.shape[1]
	channels = 1
	if _type == 'rgb':
		channels = 3
	classes = nb_classes
	index = range(0, len(X_train))
	random.shuffle(index)
	X = np.zeros((size, pixels, pixels, channels))
	Y = np.zeros((size, classes))
	for i in range(size):
		X[i] = X_train[index[i]]
		Y[i] = Y_train[index[i]]
	return X, Y

def parseSemi(X_train_semi_prime, Y_train_semi_prime, threshold=0.8, classes=10):
	X_beyond = []
	Y_beyond = []
	print 'Y_train_semi_prime[5]', Y_train_semi_prime[5], np.amax(Y_train_semi_prime[5])
	print 'Y_train_semi_prime[15]', Y_train_semi_prime[15], np.amax(Y_train_semi_prime[15])
	for i in range(len(Y_train_semi_prime)):
		max_val = np.amax(Y_train_semi_prime[i])
		if max_val >= threshold:
			X_beyond.append(X_train_semi_prime[i])
			Y_beyond.append(Y_train_semi_prime[i])
	X_semi = np.zeros((len(X_beyond), X_train_semi_prime.shape[1], X_train_semi_prime.shape[2], X_train_semi_prime.shape[3]))
	Y_semi = np.zeros((len(Y_beyond), classes))
	for i in range(len(X_beyond)):
		X_semi[i] = X_beyond[i]
		Y_semi[i] = Y_beyond[i]
	return (X_semi, Y_semi)

def parseAuto(X_train_auto_prime, Y_train_auto_prime, predict, threshold=0.8, classes=10):
	X_beyond = []
	Y_beyond = []
	print 'predict[5]', predict[5], np.amax(predict[5])
	print 'predict[15]', predict[15], np.amax(predict[15])
	for i in range(len(predict)):
		max_val = np.amax(predict[i])
		if max_val >= threshold:
			X_beyond.append(X_train_auto_prime[i])
			print i, predict[i]
			Y_train_auto_prime[i, np.argmax(predict[i])] = 1
			print Y_train_auto_prime[i]
			Y_beyond.append(Y_train_auto_prime[i])
	X_auto = np.zeros((len(X_beyond), X_train_auto_prime.shape[1], X_train_auto_prime.shape[2], X_train_auto_prime.shape[3]))
	Y_auto = np.zeros((len(Y_beyond), classes))
	for i in range(len(X_beyond)):
		X_auto[i] = X_beyond[i]
		Y_auto[i] = Y_beyond[i]
	return (X_auto, Y_auto)

def to_categorical(result, nb_classes):
	Y = np.zeros((len(result), nb_classes))
	for i in range(len(result)):
		Y[i, np.argmax(result[i])] = 1
	return Y

def countMean(X_train, Y_train, nb_classes):
	X = np.zeros((nb_classes, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
	tmp = []
	for i in range(nb_classes):
		tmp.append([])
	for i in range(len(Y_train)):
		tmp[np.argmax(Y_train[i])].append(X_train[i])
	for i in range(nb_classes):
		if len(tmp[i]) > 0:
			ndarr = np.zeros((len(tmp[i]), X_train.shape[1], X_train.shape[2], X_train.shape[3]))
		else:
			print "Warning: class", i, 'is empty'
			continue
		for j in range(len(tmp[i])):
			ndarr[j] = tmp[i][j]
		X[i] = np.mean(ndarr, axis=0)
	return X

def reshape(data):
	shape = data.shape
	multiply = 1
	for i in range(1, len(shape)):
		multiply *= shape[i]
	return data.reshape(shape[0], multiply)

def raw(Y_train):
	Y = np.zeros((len(Y_train), 1))
	for i in range(len(Y_train)):
		Y[i] = np.argmax(Y_train[i])
	return Y
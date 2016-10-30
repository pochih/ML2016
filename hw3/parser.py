import numpy as np
import random

# [0-9][0-499][0-3071] to (5000, 32, 32, 3), (5000, 10)
def parseTrain(data, _type='b/w'):
	pixels = int(len(data[0][0]) ** 0.5)
	channels = 1
	if _type == 'rgb':
		pixels = int((len(data[0][0]) / 3) ** 0.5)
		channels = 3
	classes = len(data)
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

def parseUnlabel(data, _type='b/w'):
	pixels = int(len(data[0]) ** 0.5)
	channels = 1
	if _type == 'rgb':
		pixels = int((len(data[0]) / 3) ** 0.5)
		channels = 3
	classes = 10
	data_len = len(data)
	X = np.zeros((data_len, pixels, pixels, channels))
	Y = np.zeros((data_len, classes))
	for i in range(data_len):
		for k in range(pixels):
			for l in range(pixels):
				index = k*pixels + l
				X[i][k][l] = [data[i][index], data[i][index+pixels*pixels], data[i][index+pixels*pixels*2]]

	return X, Y

def parseTest(data, _type='b/w'):
	pixels = int(len(data['data'][0]) ** 0.5)
	channels = 1
	if _type == 'rgb':
		pixels = int((len(data['data'][0]) / 3) ** 0.5)
		channels = 3
	classes = 10
	data_len = len(data['data'])
	X = np.zeros((data_len, pixels, pixels, channels))
	Y = np.zeros((data_len, classes))
	for i in range(data_len):
		for k in range(pixels):
			for l in range(pixels):
				index = k*pixels + l
				X[i][k][l] = [data['data'][i][index], data['data'][i][index+pixels*pixels], data['data'][i][index+pixels*pixels*2]]

	return X, Y

def parseValidation(X_train, Y_train, size, _type='b/w'):
	pixels = X_train.shape[1]
	channels = 1
	if _type == 'rgb':
		channels = 3
	classes = 10
	index = range(0, len(X_train))
	random.shuffle(index)
	X = np.zeros((size, pixels, pixels, channels))
	Y = np.zeros((size, classes))
	for i in range(size):
		X[i] = X_train[index[i]]
		Y[i] = Y_train[index[i]]
	return X, Y
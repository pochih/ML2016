import numpy as np
import random
import re
import cPickle as pickle
from decimal import Decimal

def clean_value(data):  # parse digit and dot in original data
  for i in range(len(data)):
    for j in range(len(data[i])):
      if isinstance(data[i][j], int) or isinstance(data[i][j], float):
        data[i][j] = float(data[i][j])
      else:
        value = re.findall("[\d.]*\d+", data[i][j])
        if len(value) == 0:
          value = 0
        else:
          value = float(value[0])
        data[i][j] = value

  return data

# parse raw training data (include data that cross days)
raw_data = [[] for x in range(18)]
with open('train.csv', 'r') as f:
  next(f)
  count = 0
  for line in f:
    line = line.split(',')
    raw_data[count%18] += line[3:]
    count += 1

raw_data = clean_value(raw_data)

# make training data
attr_len = 24*20*12  # hours*day*month
X_train = []
Y_train = []
for i in range(attr_len-9):
  if i % (20*24) > (20*24-10):
    continue
  X_tmp = [1]
  for index in range(18):
    X_tmp += raw_data[index][i:i+9]
  X_train.append(X_tmp)
  Y_train.append(raw_data[9][i+9])  # PM2.5 value
X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)
print X_train[0],X_train[-1],len(X_train)
print Y_train[0],Y_train[-1],len(Y_train)

# linear regression
epoches = 300000
batch_size = 256
learning_rate = 0.0000000105
LRate = '%.2E' % Decimal(learning_rate)
parameters = np.random.randn(163)
for i in range(epoches):
  if i % 25000 == 24999:
    learning_rate /= 1.1
  sgd_index = random.sample(range(len(X_train)), len(X_train))
  for j in range(len(X_train)/batch_size):
    # print "epoch %d, batch %d" % (i, j)
    X_batch = np.array([X_train[x] for x in sgd_index[batch_size*j:batch_size*(j+1)]])
    Y_batch = np.array([Y_train[x] for x in sgd_index[batch_size*j:batch_size*(j+1)]])
    # print 'X_batch.shape',X_batch.shape,'Y_batch.shape',Y_batch.shape
    # matrix multiply (1*128) dot (128*163) = (1*163)
    gradient = np.dot((Y_batch - np.dot(X_batch, parameters)).T, (-1)*X_batch)
    # print 'gradient.shape',gradient.shape,'parameters.shape',parameters.shape
    parameters -= learning_rate*gradient
  # matrix multiply (N*163) dot (163*1) = (N*1)
  error = 0.5*((Y_train - np.dot(X_train, parameters))**2)
  print 'Epoch %d, Error %f, LRate %.2e' % (i, np.sum(error), Decimal(learning_rate))

# parse testing data
X_test = []
with open('test_X.csv', 'r') as f:
  count = 0
  for line in f:
    if count % 18 == 0:
      X_test.append([1])
    line = line.split(',')
    X_test[-1] += line[2:]
    count += 1

X_test = clean_value(X_test)
X_test = np.array(X_test, dtype=np.float32)
print X_test[0],X_test[-1],len(X_test)

# predict
parameters_T = parameters.T
output_file = "submit_epoch-%d_batch-%d_LRate-%s" % (epoches, batch_size, LRate)
with open(output_file, 'w') as f:
  f.write("id,value\n")
  for i in range(len(X_test)):
    # matrix multiply (1*163) dot (163*1) = (1*1)
    predict = np.dot(parameters_T, X_test[i])
    f.write("id_%d,%f\n" % (i, predict))

parameter_file = "parameters_epoch-%d_batch-%d_LRate-%s" % (epoches, batch_size, LRate)
pickle.dump(parameters, open(parameter_file,'wb'), True)
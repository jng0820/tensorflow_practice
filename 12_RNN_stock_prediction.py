import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


input_dimension = 5
sequence_length = 7
output_dimension = 1
hidden_dim = 10

def MinMaxScalar(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


xy = np.loadtxt('data-02-stock_daily.csv',delimiter=',')
xy = xy[::-1]
xy = MinMaxScalar(xy)
x = xy
y = xy[:,[-1]]

dataX = []
dataY = []

for i in range(0, len(y) - sequence_length):
    _x = x[i:i+sequence_length]
    _y = y[i+sequence_length]
    print(_x, "->", _y)

    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY)*0.7)
test_size = len(dataY) - train_size

trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32, shape=[None, sequence_length, input_dimension])
Y = tf.placeholder(tf.float32, shape=[None, 1])

cell = rnn.BasicLSTMCell(num_units = hidden_dim,state_is_tuple=True)
outputs, _state = tf.nn.dynamic_rnn(cell, X,dtype=tf.float32)
Y_predict = tf.contrib.layers.fully_connected( outputs[:,-1],output_dimension,activation_fn = None)

loss = tf.reduce_sum(tf.square(Y_predict-Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, train_loss = sess.run([optimizer,loss],feed_dict={X:dataX,Y:dataY})

    print(i,train_loss)



test_predict = sess.run(Y_predict,feed_dict={X:testX})

import matplotlib.pyplot as plt

plt.plot(testY)
plt.plot(test_predict)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()

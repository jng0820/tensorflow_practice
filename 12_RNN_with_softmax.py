import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i ,w in enumerate(char_set)}

dataX = []
dataY = []
seq_length = 10

for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i+1: i+ seq_length+1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)

batch_size = len(dataX)


X = tf.placeholder(tf.int32,shape=[None,seq_length])
Y = tf.placeholder(tf.int32,shape=[None,seq_length])
X_one_hot = tf.one_hot(X, num_classes)

cell = rnn.BasicLSTMCell(hidden_size,state_is_tuple= True)
cell = rnn.MultiRNNCell([cell] * 6, state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot,dtype=tf.float32)

X_for_softmax = tf.reshape(outputs,[-1,hidden_size])
softmax_w = tf.get_variable("softmax_w",[hidden_size,num_classes])
softmax_b = tf.get_variable("softmax_b",[num_classes])

outputs = tf.matmul(X_for_softmax,softmax_w) + softmax_b
outputs = tf.reshape(outputs,[batch_size,seq_length,num_classes])

weights = tf.ones([batch_size, seq_length])

sequence_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=outputs, targets = Y, weights = weights))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(sequence_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(500):
    _, l, results = sess.run([optimizer,sequence_loss,outputs],feed_dict={X:dataX,Y:dataY})

    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

results = sess.run(outputs, feed_dict={X: dataX})

for j, result in enumerate(results):
    index = np.argmax(result,axis=1)
    if j is 0:
        print(''.join([char_set[c] for c in index]),end='')
    else:
        print(char_set[index[-1]],end='')
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

idx2char = [ 'h','i','e','l','o']
h = [1,0,0,0,0]
i = [0,1,0,0,0]
e = [0,0,1,0,0]
l = [0,0,0,1,0]
o = [0,0,0,0,1]
x_data = [[h,i,h,e,l,l]]
y_data = [[1,0,2,3,3,4]]

rnn_size = 5
hidden_size =5
input_dim = 5
batch_size = 1
sequence_length = 6

X = tf.placeholder(tf.float32,shape=[None,sequence_length,input_dim])
Y = tf.placeholder(tf.int32,shape=[None,sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size,state_is_tuple = True)
initial_state = cell.zero_state(batch_size,tf.float32)

outputs, _states = tf.nn.dynamic_rnn(cell,X,initial_state=initial_state,dtype=tf.float32)
weights = tf.ones([batch_size,sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets=Y,weights = weights)
# 원래는 rnn에 나온걸 바로 logits에 넣으면 좋지 않음

loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs,axis=2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2000):
    l, _ = sess.run([loss,train], feed_dict={X:x_data,Y:y_data})
    result = sess.run(prediction,feed_dict={X:x_data})
    print(i, "loss : " , l , "prediction : ", result, "True Value : ", y_data)

    result_str = [idx2char[c] for c in np.squeeze(result)]
    print("\tPrediction str : ", ''.join(result_str))

import tensorflow as tf
import numpy as np
import pprint
from tensorflow.contrib import rnn

# 출력의 size
hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

x_data = np.array([[h, e, l, l, o],[e,o,l,l,l],[l,l,e,e,l]], dtype=np.float32)
print(x_data.shape)

outputs, _state = tf.nn.dynamic_rnn(cell,x_data,dtype=tf.float32)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(x_data)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(outputs.shape)
pp.pprint(outputs.eval())

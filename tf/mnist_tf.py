#%%
from __future__ import division
import numpy as np
import pandas as pd

#%%
# dataset
train = pd.read_csv("/Users/edouardcuny/Downloads/train.csv")
x = train.iloc[:,1:]
y = train.iloc[:,0]
x = x.as_matrix()
x= x/255
y = y.as_matrix()
x = x.astype('float64')
y = y.astype('float64')

#%%
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#%%

# VARIABLES
x = tf.placeholder(tf.float32, [None, 784])
h = tf.placeholder(tf.float32, [None, 30])
W1 = tf.Variable(tf.random_normal([784, 30]))
W2 = tf.Variable(tf.random_normal([30, 10]))
b1 = tf.Variable(tf.zeros([30]))
b2 = tf.Variable(tf.zeros([10]))

# FEEDFORWARD
z1 = tf.matmul(x, W1) + b1
h = tf.sigmoid(z1)
z2 = tf.matmul(h, W2) + b2
y = z2              
y = tf.sigmoid(z2)
             
# LOSS & OPTIMIZER
y_ = tf.placeholder(tf.float32, [None, 10])

quadratic_cost = tf.losses.mean_squared_error(labels=y_, predictions=y)
tf.summary.scalar('erreur_quadratique', quadratic_cost)

train_step = tf.train.GradientDescentOptimizer(3).minimize(quadratic_cost)

# SESSION TENSORFLOW
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# SUMMARIES FOR THE TENSORBOARD
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/Users/edouardcuny/Desktop/train',
                                      sess.graph)


# TRAIN
batch_size = 10
epochs = 30
train_size = 6000 # on le suppose

for i in range(int((epochs*train_size)/batch_size)):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      
# TEST
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))    

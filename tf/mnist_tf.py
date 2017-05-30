#%%
from __future__ import division
import numpy as np
import pandas as pd
import tensorflow as tf

#%%

# TRAIN
train = pd.read_csv("/Users/edouardcuny/Downloads/train.csv")
x_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
x_train = x_train.as_matrix()
x_train = x_train/255

y_train = y_train.as_matrix()
x_train = x_train.astype('float64')
y_train = y_train.astype('float64')

# y_train from labels to one hot
y = np.zeros([y_train.shape[0],10])
for i in range(y.shape[0]):
    y[i, int(y_train[i])]=1
y_train = y


# TEST
test = pd.read_csv("/Users/edouardcuny/Downloads/test.csv")
x_test = test.as_matrix()
x_test = x_test/255





#%%
def next_batch_x_train(batch_size):
    global index_batch_x_train
    array = x_train
    
    if batch_size > array.shape[0]:
        raise IndexError
    
    if (index_batch_x_train+1)*batch_size > array.shape[0]:
        index = index_batch_x_train
        index_batch_x_train = 0
        return array[index*batch_size:,:]
    
    else:
        index = index_batch_x_train
        index_batch_x_train += 1
        return array[index*batch_size:(index+1)*batch_size]

def next_batch_y_train(batch_size):
    global index_batch_y_train
    array = y_train
    
    if batch_size > array.shape[0]:
        raise IndexError
    
    if (index_batch_y_train+1)*batch_size > array.shape[0]:
        index = index_batch_y_train
        index_batch_y_train = 0
        return array[index*batch_size:,:]
    
    else:
        index = index_batch_y_train
        index_batch_y_train += 1
        return array[index*batch_size:(index+1)*batch_size]

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

index_batch_x_train = 0
index_batch_y_train = 0
for i in range(int((epochs*train_size)/batch_size)):
  batch_xs = next_batch_x_train(batch_size)
  batch_ys = next_batch_y_train(batch_size)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      
# TEST
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_train,
                                    y_: y_train}))

#%% TRAINING IS NOW DONE
prediction = sess.run(tf.argmax(y,1), feed_dict={x: x_test})
prediction = pd.DataFrame(prediction)
prediction[1] = [x+1 for x in range(prediction.shape[0])]
prediction = prediction.iloc[:,[1,0]]
prediction.columns = ['ImageId', 'Label']
prediction.to_csv('/Users/edouardcuny/Desktop/ml/tf/submission.csv', index=False)

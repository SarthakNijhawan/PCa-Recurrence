import tensorflow as tf
import numpy as np
from keras.layers import Dense, Dropout, Activation
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras import backend as K

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./tmp/MNIST_data', one_hot=True)

# Global Variables
n_epochs = 5

sess = tf.Session()
K.set_session(sess)

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

# Keras layers can be called on TensorFlow tensors:
x = Dense(128, activation=None)(img)  # fully-connected layer with 128 units and ReLU activation
# x = tf.nn.relu(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation=None)(x)
# x = tf.nn.relu(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation

# loss funtion
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# Training operation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop for one epoch
with sess.as_default():
	for ep in range(n_epochs):
	    for i in range(100):
	        batch = mnist_data.train.next_batch(50)
	        train_step.run(feed_dict={img: batch[0],
	                                  labels: batch[1],
	                                  K.learning_phase(): 1})
	    print("Epoch : " + str(ep+1) +" loss is : " + loss.eval(sess, feed_dict={img: batch[0],
												                                 labels: batch[1],
												                                 K.learning_phase(): 0}))
# Accuracy metric
acc_value = tf.reduce_mean(accuracy(labels, preds))
with sess.as_default():
    print(acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0}))

# Saving the model
saver = tf.train.Saver()
saver.save(sess, './tmp/models/mnist_ff_keras', global_step=1)								# Let say global_step are the number of epochs the model has gone through

# Load a model and predict
# predictions = preds.eval(feed_dict={img: img_pred,
# 									K.learning_phase(): 0})

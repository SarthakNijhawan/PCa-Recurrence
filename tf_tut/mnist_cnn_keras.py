import tensorflow as tf
import numpy as np
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras import backend as K

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./tmp/MNIST_data', one_hot=True)

sess = tf.Session()
K.set_session(sess)

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

# Keras layers can be called on TensorFlow tensors:
input_layer = tf.reshape(img, [-1, 28, 28, 1])
conv1 = Conv2D(32, 5, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(input_layer)
conv1 = tf.nn.local_response_normalization(conv1)										# FIXME
conv1 = Activation('relu')(conv1)
conv1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

conv2 = Conv2D(64, 5, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(conv1)
conv2 = tf.nn.local_response_normalization(conv2)										# FIXME
conv2 = Activation('relu')(conv2)
conv2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

conv2_flatten = tf.reshape(conv2, [-1, 7*7*64])

dense1 = Dense(1024, activation='relu')(conv2_flatten)
dense1 = Dropout(0.8)(dense1)

preds = Dense(10, activation='softmax')(dense1)

# loss funtion
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# Training operation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop for one epoch
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1],
                                  K.learning_phase(): 1})
# Accuracy metric
acc_value = tf.reduce_mean(accuracy(labels, preds))
with sess.as_default():
    print(acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0}))

# # Prediction for a given batch of images
# predictions = preds.eval(feed_dict={img: img_pred,
# 									K.learning_phase(): 0})

import cv2
import numpy as np
import tensorflow as tf

from keras.layers import Dense, Dropout, Activation
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras import backend as K

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./tmp/MNIST_data', one_hot=True)

#------------------------------------------ Global Variables
n_epochs = 5

sess = tf.Session()
K.set_session(sess)

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

#------------------------------------------ Architecture
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

pred_class = tf.argmax(preds, axis=1)

# loss funtion
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# Training operation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

#------------------------------------------ Run training loop for one epoch
with sess.as_default():
	for ep in range(n_epochs):
	    for i in range(100):
	        batch = mnist_data.train.next_batch(50)
	        train_step.run(feed_dict={img: batch[0],
	                                  labels: batch[1],
	                                  K.learning_phase(): 1})

	    print("Epoch : " + str(ep+1) + " loss is : " + str(loss.eval(feed_dict={img: batch[0],
												                                 labels: batch[1],
												                                 K.learning_phase(): 0})))
#------------------------------------------ Accuracy metric
acc_value = tf.reduce_mean(accuracy(labels, preds))
with sess.as_default():
    print(acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0}))

#------------------------------------------ Saving the model
saver = tf.train.Saver()
saver.save(sess, './tmp/models/mnist_ff_keras/model_ff_keras')								# Let say global_step are the number of epochs the model has gone through

#------------------------------------------ Load a model and predict
loaded_img = cv2.imread("img.jpg", 0)
loaded_img = np.reshape(loaded_img, [1, 784])

saver.restore(sess, './tmp/models/mnist_ff_keras/model_ff_keras')
print(sess.run([preds, pred_class], feed_dict={img: loaded_img,
									K.learning_phase(): 0}))

#------------------------------------------ Aliter by loading the graph (.meta file) [Not working though :(, just decomment it ]
# load meta graph and restore weights
# saver = tf.train.import_meta_graph('./tmp/models/mnist_ff_keras/model_ff_keras.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./tmp/models/mnist_ff_keras/'))



# graph = tf.get_default_graph()
# # print(graph.get_operation_by_name("img:)"))
# img = graph.get_tensor_by_name("img:0")														# Placeholder for prediction

# # access the op
# preds = graph.get_tensor_by_name("preds:0")
# pred_class = graph.get_tensor_by_name("pred_class:0")


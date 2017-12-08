from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2

# Increases verbosity in the log
tf.logging.set_verbosity(tf.logging.INFO)

def patch_base_cnn_model_fn(features, labels, mode):
	""" Model function for patch based CNN """

	input_layer = tf.reshape(features["x"], [-1, 101, 101, 3])

# ------------------ Layer1 -------------------------
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=80,
		kernel_size=[10,10],
		strides=(2,2),
		padding="same",
		activation=tf.nn.relu)

	lrn1 = tf.nn.local_respose_normalsation(
		inputs=conv1,
		depth_radius=5,
		bias=1,
		alpha=1,
		beta=0.5,
		name=None)

	pool1 = tf.layers.max_pooling2d(
		inputs=lrn1,
		pool_size=[2,2],
		strides=2)

# ------------------ Layer2 -------------------------
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=80,
		kernel_size=[10,10],
		strides=(2,2),
		padding="same",
		activation=tf.nn.relu)

	lrn2 = tf.nn.local_respose_normalsation(
		inputs=conv2,
		depth_radius=5,
		bias=1,
		alpha=1,
		beta=0.5,
		name=None)

	pool2 = tf.layers.max_pooling2d(
		inputs=lrn2,
		pool_size=[2,2],
		strides=2)

# ------------------ Layer3 -------------------------
	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=80,
		kernel_size=[10,10],
		strides=(2,2),
		padding="same",
		activation=tf.nn.relu)

# ------------------ Layer4 -------------------------
	conv4 = tf.layers.conv2d(
		inputs=conv3,
		filters=80,
		kernel_size=[10,10],
		strides=(2,2),
		padding="same",
		activation=tf.nn.relu)

	pool4 = tf.layers.max_pooling2d(
		inputs=conv4,
		pool_size=[2,2],
		strides=2)

# ------------------ Dense Layer 1-------------------------
	pool4_flat = tf.reshape(pool4, [-1, ])
	dense_layer1 = tf.layers.dense(
		inputs=pool4_flat,
		units=1024,
		activation=tf.nn.relu)

	dropout1 = tf.layers.dropout(
		inputs=dense_layer1,
		rate=0.5,
		training=mode==tf.estimator.ModeKeys.TRAIN)

# ------------------ Dense Layer 2-------------------------
	dense_layer2 = tf.layers.dense(
		inputs=dropout1,
		units=1024,
		activation=tf.nn.relu)	

	dropout2 = tf.layers.dropout(
		inputs=dense_layer2,
		rate=0.5,
		training=mode==tf.estimator.ModeKeys.TRAIN)

# ------------------ Logits Layer -------------------------
	logits = tf.layers.dense(
		inputs=dropout2,
		units=1,
		activation=None)

def main();
	""" Main function """


	# Create the Estimator
	patch_based_cnn_classifier = tf.estimator.Estimator(
	    model_fn=cnn_model_fn, model_dir="./tmp/mnist_convnet_model")

	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True)

	for i in range(12):
		patch_based_cnn_classifier.train(
			input_fn=train_input_fn,
			num_epochs=2,
			hooks=[logging_hook])


if __name__=="__main__":
	tf.app.run()

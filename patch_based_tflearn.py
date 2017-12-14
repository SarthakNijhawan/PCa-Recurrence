import cv2
import numpy as np
import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.normalisation import local_response_normalisation
from tflearn.layers.estimator import regression


# Global Variables
keep_rate = 0.5
n_classes = 2
batch_size=64

def model_cnn(keep_prob, batch_size):
	""" Model function for CNN """
	input_layer = input_data(shape=[-1,101,101,3], name="input")								# Since patch_size = 101x101x3

	conv1 = conv_2d(input_layer, 80, [6,6], strides=1, padding='same', activation='relu')
	conv1 = local_response_normalisation(conv1)
	conv1 = max_pool_2d(conv1, [2,2], 2)

	conv2 = conv_2d(conv1, 120, [5,5], strides=1, padding='same', activation='relu')
	conv2 = local_response_normalisation(conv2)
	conv2 = max_pool_2d(conv2, [2,2], 2)

	conv3 = conv_2d(conv2, 160, [3,3], strides=1, padding='same', activation='relu')

	conv4 = conv_2d(conv3, 200, [3,3], strides=1, padding='same', activation='relu')
	conv4 = max_pool_2d(conv4, [3,3], 2)

	dense1 = fully_connected(conv4, 320, activation='relu')
	dense1 = dropout(dense1, keep_prob)

	dense2 = fully_connected(dense1, 320, activation='relu')
	dense2 = dropout(dense2, keep_prob)

	output = fully_connected(dense2, n_classes, activation='softmax')							# Output layer with predicion probabilites
	
	convnet = regression(output, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', 
						 name='targets', batch_size=batch_size, n_classes=n_classes, metric='accuracy')

	return [output, convnet]

def train_and_save_cnn(X, Y, val_x, val_y, keep_prob=0.5, n_epochs=2, batch_size=32):
	""" Trains the model for given number of epochs and then saves the model """
	_, convnet = model_cnn(keep_prob, batch_size)

	model = tflearn.DNN(convnet, tensorboard_dir='./tmp/tflearn_logs/')
	model.fit({'input': X}, {'targets': Y}, n_epoch=n_epochs, validation_set=({'input': val_x}, {'targets': val_y}), 
	    		snapshot_step=500, show_metric=True, run_id='patch_based_cnn')

	# model.fit({'input': X}, {'targets': Y}, n_epoch=n_epochs, validation_set=0.1, 			# Splits the data for 10% val_set
	#     		snapshot_step=500, show_metric=True, run_id='patch_based_cnn')
	
	# save the model
	model.save('./tmp/models/patch_based_cnn.model')

def load_and_predict(pred_x, keep_prob=0.5, batch_size):
	""" Loads the trained model and predicts """
	output, convnet = model_cnn(keep_prob, batch_size)
	model = tflearn.DNN(convnet, tensorboard_dir='./tmp/tflearn_logs/')
	model.load('./tmp/models/patch_based_cnn.model')

	prediction = model.predict(pred_x)															# Predictions from the model
	return 

def perform_gaussian_smoothing():
	""" Perform gaussian smoothing on the predictions and 
		generates a bit mask 
	"""


	pass

def update_training_data():
	""" 
	"""

	pass

def main(n_iter):
	# EM Algo is performed in this loop
	data = {
		"patches":
		"labels"
	}

	for it in range(n_iter):
		train_and_save_cnn()												# runs the model for 2 epochs for the given training data
		load_and_predict()
		perform_gaussian_smoothing()
		update_training_data()

if __name__ == '__main__':
	main(0)
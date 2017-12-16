import argparse																						#TODO
import numpy as np
import os
import pickle
import scipy.stats as st
import tensorflow as tf

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras import backend as K

# --------------------- Global Variables -------------------------
model_path = './models/patch_based_cnn_model'
patches_path = '../deepak/DB_HnE_101_anno_cent'
images_path = '../deepak/sorted_patient_wise_normalized_dataset'

n_iter = 1000														# number of iterations for EM algo to run

# --------------------- Main Function ----------------------------
def main():
	""" Trains the model on EM algo and saves it to the given model path """

	# Loads all the patches label wise dictionaries of image wise patches 
	patches, initial_train_data = load_patches_image_wise(patches_path, images_path)				# TODO

	# Initial M-step
	_, _, _, train_step = patch_based_cnn_model()
	train(initial_train_data[0], initial_train_data[1], load=False)

	for itr in range(n_iter-1):
		# E-step
		predicted_maps = []
		for i in range(2):
			per_label_maps = {}
			for image in patches[i].keys:
				per_label_maps{image} = load_and_predict(patches{image})							# patches{image}.shape = [None, 101, 101, 3]
			predicted_maps[i] = per_label_maps
		train_data = E_step(predicted_maps)															# TODO

		# M-Step
		train(train_data[0], train_data[1], model_path)
		# load_and_evaluate(test_data[0], test_data[1], model_path)									# TODO
	
	print("Completed!!")


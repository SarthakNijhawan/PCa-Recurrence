import argparse														#TODO
import numpy as np
import os
import pickle
import tensorflow as tf

"""
	Description:

"""

# Global Variables
dropout_prob = 0.5													# Dropout layer probability
n_epochs = 2														# Number of epochs cnn must be trained in each M step
n_classes = 2														# Recurrent and Non-recurrent 
batch_size = 128													# batch size for training the cnn
percentile_1 = 0.4													# FIXME
percentile_2 = 0.3													# FIXME
n_iter = 1000														# number of iterations for EM algo to run

def load_image(img_path):
	img = cv2.imread(img_path)
	img_tensor = tf.convert_to_tensor(img)
	return tf.reshape(img_tensor, [1,101,101,3])					# Reshaping so that it can be processed further into batches on axis=0

def load_imagewise_patches():
	""" - Loads all the images and splits them into patches for prediction
		and updation of training data for every iteration in the applied
		EM algo for training he model
		- Returns:
			All the patches arranged in a dict img_wise (will be required when reconstructin the probability map)

	"""
	cent_patches_dir = '../deepak/DB_HnE_101_anno_cent'
	normalised_dataset_dir = '../deepak/sorted_patient_wise_normalized_dataset'

	img_wise_patches = []	
	for i in range(2):												# iterated over label 1 and 0
		labeled_img_wise_patches = {}
		patient_wise_image_names = {}								# dict with img patient as keys and list of associated image names

		labeled_images_dir = os.path.join(normalised_dataset_dir, 'label_'+str(i))		
		patient_wise_images_dirlist = os.listdir(labeled_images_dir)
		for patient_dirname in patient_wise_images_dirlist:
			patient_id = patient_dirname.split("_")[0]
			patient_wise_image_names{patient_id} = []

			patient_dir_path = os.path.join(labeled_images_dir, patient_dirname)
			file_list = os.listdir(patient_dir_path)
			for filename in file_list:
				if file.endswith(".xml"):
					patient_wise_image_names{patient_id} += [filename.split(".")[0],]
					labeled_img_wise_patches{filename.split(".")[0]} = []			# initialises every dict element for an image to an empty list

		labeled_patches_dir = os.path.join(cent_patches_dir, 'label_'+str(i))
		patient_wise_patches_dirlist = os.listdir(labeled_patches_dir)
		for patient_dirname in patient_wise_patches_dirlist:
			patient_id = patient_img_dirname.split("_")[0]			# for every patient image
			patient_dir_path = os.path.join(labeled_patches_dir, patient_dirname)
			patch_list = os.listdir(patient_dir_path)
			for patchname in patch_list:
				patch_path = os.path.join(patient_dir_path, patchname)
				img_name = patchname.split("_")[-1].split(".")[0]
				labeled_img_wise_patches{img_name} += [load_patch(patch_path),]

		img_wise_patches += [labeled_img_wise_patches,] 

	return img_wise_patches


def prepare_train_data_from_img_wise_dict():
	""" - Loads the initially extracted nucleus centric patches for training 
		and treating all the patches as discrminiative initially
		- Returns:

		"""

	pass

def M_step():
	""" - Trains the model with discrminative patches for n_epochs = 2 (here), 
		and saves the model, useful in predicting the probability maps per image 
		for the next E-step
		- Returns:

		"""

	pass

def load_and_predict():
	"""	- Loads the model and predicts recurrence behaviour for a 2D list(array) 
		of images, split into patches and assembles their output into an 2-D for 
		further processing 
		- Returns:

	"""

	pass

def E_step():
	""" - Applies gaussian smoothing to the predicted probability maps
		- Generates bit mask for acceptable()discriminative patches for the next M_step
		- Returns:
				Refined training data for the next step
	"""
	
	pass


def main():
	""" Trains the model on EM algo and saves it """
	imagewise_patches = load_imagewise_patches() 							# Required in each iteration of EM algo to refine the training data
	training_data = prepare_train_data_from_img_wise_dict(imagewise_patches)

	for itr in range(n_iter):
		
		# trains and save the model based on the given discriminative patches
		M_step(training_data, n_epochs, batch_size, ):

		# Predicts the probab maps and then performs the e-step
		predicted_probability_maps = load_model_and_predict(imagewise_patches)

		training_data = E_step(predicted_probability_maps)

		# Check when to stop @todo

if __name__ == '__main__':
	main()

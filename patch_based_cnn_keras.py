import argparse														#TODO
import numpy as np
import os
import pickle
import scipy.stats as st
import tensorflow as tf

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras import backend as K

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


# Default Session for Keras
sess = tf.Session()
K.set_session(sess)

# -------------------- Helper functions ------------------------
def load_image(img_path):
	img = cv2.imread(img_path)
	img_tensor = tf.convert_to_tensor(img)
	return tf.reshape(img_tensor, [1,101,101,3])					# Reshaping so that it can be processed further into batches on axis=0

# local response normalization on images (Adapted Code)
# def lrn(x):
# 	""" Custom made normalisation layer for Keras """
#     ones_for_weight = np.reshape(np.ones((32, 32)), (1,32,32))
#     mu = sum_pool2d(x, pool_size = (7, 7), strides = (1, 1), padding = (3, 3))
#     mu_weight = sum_pool2d(ones_for_weight, pool_size = (7, 7), strides = (1, 1), padding = (3, 3))
#     sum_sq_x = sum_pool2d(K.square(x), pool_size = (7, 7), strides = (1, 1), padding = (3, 3))
#     total_mu_sq = mu_weight * K.square(mu)
#     sq_cross_term = -2 * K.square(mu)
#     sigma = K.sqrt(sum_sq_x + total_mu_sq + sq_cross_term)
#     return (x - mu)/(sigma + 1)

# # def lcn_output_shape(input_shape):
# #     return input_shape


# def sum_pool2d(x, pool_size = (7, 7), strides = (1, 1), padding = (3, 3)):
#     sum_x = pool.pool_2d(x, ds = pool_size, st = strides, mode = 'sum', padding = padding, ignore_border = True)
#     return sum_x

# # def sum_pool2d_output_shape(input_shape):
# #     return input_shape

def gkern(kernlen=101, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def cnn_model():
	""" Model function for the CNN Model """
	# Placeholders
	img = tf.placeholder(tf.float32, shape=(None, 101, 101, 3))
	labels = tf.placeholder(tf.float32, shape=(None, 2))

	# Layers
	conv1 = conv2D(80, 6, strides=1, padding='same', activation=None, kernel_initialiser='he_normal')(img)
	conv1 = tf.nn.local_response_normalisation(conv1)										# FIXME
	conv1 = Activation('relu')(conv1)
	conv1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

	conv2 = conv2D(120, 5, strides=1, padding='same', activation=None, kernel_initialiser='he_normal')(conv1)
	conv2 = tf.nn.local_response_normalisation(conv2)										# FIXME
	conv2 = Activation('relu')(conv2)
	conv2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

	conv3 = conv2D(160, 3, strides=1, padding='same', activation=None, kernel_initialiser='he_normal')(conv2)
	
	conv4 = conv2D(200, 3, strides=1, padding='same', activation=None, kernel_initialiser='he_normal')(conv3)
	conv4 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv4)

	dense1 = Dense(320, activation='relu')(conv4)
	dense1 = Dropout(dropout_prob)(dense1)

	dense2 = Dense(320, activation='relu')(dense1)
	dense2 = Dropout(dropout_prob)(dense2)

	output_softmax = Dense(320, activation='relu')(dense2)

	# loss funtion
	loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

	# Training operation
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	return [output_softmax, loss, train_step]

# -------------------- Functions for EM Algo -------------------
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
	[output, loss, train_step] = cnn_model()

	# Initialize all variables
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	# Run training loop
	with sess.as_default():
	    for i in range(100):
	        batch = mnist_data.train.next_batch(50)
	        train_step.run(feed_dict={img: batch[0],
	                                  labels: batch[1],
	                                  K.learning_phase(): 1})
	# Accuracy metric
	acc_value = accuracy(labels, preds)
	with sess.as_default():
	    print acc_value.eval(feed_dict={img: mnist_data.test.images,
	                                    labels: mnist_data.test.labels,
	                                    K.learning_phase(): 0})

	# Prediction for a given batch of images
	predictions = preds.eval(feed_dict={img: img_pred,
										K.learning_phase(): 0})

	pass

def load_and_predict():
	"""	- Loads the model and predicts recurrence behaviour for a 2D list(array) 
		of images, split into patches and assembles their output into an 2-D for 
		further processing 
		- Returns:

	"""

	pass

def E_step(predicted_maps):
	""" - Applies gaussian smoothing to the predicted probability maps
		- Generates bit mask for acceptable()discriminative patches for the next M_step
		- Returns:
				Refined training data for the next step
	"""

	for labeled_dict in predicted_maps:
		for img_name in labeled_dict.keys:
			reconstructed_probab_map = np.zeros([101,101])
			
			for patch_tuple in labeled_dict[img_name]:
				center_coord = patch_tuple[:2]
				probability = patch_tuple[2]
				orig_patch_probab = reconstructed_probab_map[center_coord[0]-50: center_coord[0]+50, center_coord[1]-50:center_coord[1]+50]
				new_patch_prob = probability*np.gkern()
				reconstructed_probab_map[center_coord[0]-50: center_coord[0]+50, center_coord[1]-50:center_coord[1]+50] = 
						np.maximum(orig_patch, new_patch_prob)

			
			threshold = 






	pass


def main():
	""" Trains the model on EM algo and saves it """
	imagewise_patches = load_imagewise_patches() 							# Required in each iteration of EM algo to refine the training data
	training_data = prepare_train_data_from_img_wise_dict(imagewise_patches)

	for itr in range(n_iter):
		
		# trains and save the model based on the given discriminative patches
		M_step(training_data, n_epochs, batch_size, ):

		# Predicts the probab maps and then performs the e-step
		predicted_probability_maps = load_and_predict(imagewise_patches)

		training_data = E_step(predicted_probability_maps)

		# Check when to stop @todo

if __name__ == '__main__':
	main()

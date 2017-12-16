import numpy as np
import os
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

# @todo : Evaluation in every iteration
# @todo : 

# --------------------- Main Function ----------------------------
def main():
	""" Trains the model on EM algo and saves it to the given model path """
	# Default Session for Keras
	sess = tf.Session()
	K.set_session(sess)

	saver = tf.train.Saver()

	# Loads all the patches label wise dictionaries of image wise patches 
	patches, initial_train_data = load_patches_image_wise(patches_path, images_path)						# TODO

	# Initial M-step: Training and predictions of probability maps
	preds, pred_class, loss, train_step = patch_based_cnn_model()
	for epoch in xrange(1, n_epochs):
		epoch_loss = 0
		for i in range(train_x.shape[0]/batch_size):
			sess.run(train_step, feed_dict={img: train_x[,:,:,:],											# TODO
											labels: train_y[,:],											# TODO
											K.learning_phase(): 1})
		print("Epoch :", epoch, "loss is :", epoch_loss)

	# 2nd Iteration onwards
	for itr in range(n_iter-1):
		# E-step
		predicted_maps = []
		for i in range(2):
			predicted_maps[i] = {}
			for image in patches[i].keys:
				predicted_maps[i]{image} = sess.run(preds, feed_dict={	img: patches[i]{image},
																   		K.learning_phase(): 0})
		train_data = E_step(predicted_maps)																	# TODO

		# M-Step
		for epoch in xrange(1, n_epochs):
			epoch_loss = 0
			for i in range(train_x.shape[0]/batch_size):
				sess.run(train_step, feed_dict={img: train_x[,:,:,:],										# TODO
												labels: train_y[,:],										# TODO
												K.learning_phase(): 1})
			print("Epoch :", epoch, "loss is :", epoch_loss)

	# saving the model
	saver.save(sess, model_path)

	# EM-Algo completed
	sess.close()
	print("Completed!!")



##################################################################
#---------------------Model and its functions -------------------#
##################################################################
def patch_based_cnn_model(dropout_prob=0.5, l_rate=0.5, n_classes=2):

	# Placeholders
	img = tf.placeholder(tf.float32, shape=(None, 101, 101, 3))
	labels = tf.placeholder(tf.float32, shape=(None, 2))

	# Layers
	conv1 = Conv2D(80, 6, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(img)
	conv1 = tf.nn.local_response_normalisation(conv1)														# FIXME
	conv1 = Activation('relu')(conv1)
	conv1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

	conv2 = Conv2D(120, 5, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(conv1)
	conv2 = tf.nn.local_response_normalisation(conv2)														# FIXME
	conv2 = Activation('relu')(conv2)
	conv2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

	conv3 = Conv2D(160, 3, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(conv2)
	
	conv4 = Conv2D(200, 3, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(conv3)
	conv4 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv4)

	conv4_flatten = tf.reshape(conv4, [-1, 9*9*200])
	dense1 = Dense(320, activation='relu')(conv4_flatten)
	dense1 = Dropout(dropout_prob)(dense1)

	dense2 = Dense(320, activation='relu')(dense1)
	dense2 = Dropout(dropout_prob)(dense2)

	preds = Dense(n_classes, activation='softmax')(dense2)
	pred_clas = tf.argmax(preds, axis=1)

	# loss funtion
	loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

	# Training operation
	train_step = tf.train.GradientDescentOptimizer(l_rate).minimize(loss)

	# Accurace metric
	acc_value = tf.reduce_mean(accuracy(labels, preds))

	return [preds, pred_class, loss, train_step]



##################################################################
#--------------------- EM Algo Helper functions -------------------#
##################################################################
def load_patches_image_wise(cent_patches_dir, normalised_dataset_dir):
	""" - Loads all the images and splits them into patches for prediction
		and updation of training data for every iteration in the applied
		EM algo for training he model
		- Returns:
			All the patches arranged in a dict img_wise (will be required when reconstructin the probability map)

	"""

	img_wise_patches = []
	train_data = []

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

	return [img_wise_patches, init_train_data]


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
			
			# threshold = 

	return new_train_data


#################################################################
# -------------------- Other Helper functions ------------------#
#################################################################
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

if __name__ == '__main__':
	main()


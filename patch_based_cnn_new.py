import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import scipy.stats as st
import tensorflow as tf
import cv2
import shutil

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation
from keras.models import Model
from keras import backend as K


# --------------------- Global Variables -------------------------
model_path = './saved_models'
patient_wise_patches_path = './anno_cent'
data_path = './data_dump'
tmp_train_data_path = './train_tmp/'									# tmp dir for training. will be deleted b4 the program closes
val_data_path = data_path + '/valid'

n_iter = 2
batch_size = 128
n_classes = 2
data_augmentation = True

# input image dimensions
img_rows, img_cols = 101,101
# The patch images are RGB.
img_channels = 3

# @todo : Description


# --------------------- Main Function ----------------------------
def main():
	# Default Session for Keras
	sess = tf.Session()
	K.set_session(sess)

	# Maintains a copy of whole data for further reference
	shutil.copytree(os.path.join(data_path, "train"), tmp_train_data_path)

	# Defining a model
	model = patch_based_cnn_model()

	# This will do preprocessing and realtime data augmentation:
	datagen = ImageDataGenerator(rescale=1.0/255.)
	datagen_augmented = ImageDataGenerator(featurewise_center=False,
	    samplewise_center=True,
	    featurewise_std_normalization=False,
	    samplewise_std_normalization=True,
	    zca_whitening=False,
	    zca_epsilon=1e-6,
	    rotation_range=90,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    shear_range=0.2,
	    zoom_range=0.2,
	    channel_shift_range=0.2,
	    fill_mode='nearest',
	    cval=0.,
	    horizontal_flip=True,
	    vertical_flip=True,
	    rescale=1.0/255.,
	    preprocessing_function=None)
	train_generator_augmented = datagen_augmented.flow_from_directory(
	    tmp_train_data_path,
	    target_size=(img_cols, img_rows),
	    batch_size=batch_size,
	    class_mode='categorical')
	validation_generator = datagen.flow_from_directory(
	    tmp_train_data_path,
	    target_size=(img_cols, img_rows),
	    batch_size=batch_size,
	    class_mode='categorical')

	#################### Initial M-step ######################## 
	# Training and predictions of probability maps
	train(model, train_generator_augmented, validation_generator)									#TODO
	print("First iteration of EM algo over")

	#################### 2nd Iteration Onwards ########################
	for itr in range(n_iter-1):
		# E-step
		generate_predicted_maps(preds, data_path, img=img)											#TODO
		print("Probab maps predicted!!")

		E_step(data_path, tmp_train_data_path)														#TODO
		print("{}th iteration's E-step performed!!".format(itr+1))
		
		# M-Step
		train(model, train_generator_augmented, validation_generator)
		print("{}th iteration's M-step performed!!".format(itr+1))

	# saving the model
	saver = tf.train.Saver()
	saver.save(sess, model_path)

	# EM-Algo completed
	shutil.rmtree(tmp_train_data_path)
	sess.close()
	print("Training Completed Successfully !!")


##################################################################
#---------------------Model and its functions -------------------#
##################################################################
def patch_based_cnn_model(dropout_prob=0.5, l_rate=0.5, n_classes=2, img_rows=101, img_cols=101):

	# Layers
	input_img=Input(shape=(img_rows,img_cols,3))
	convnet = Conv2D(80, 6, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(input_img)
	convnet = tf.nn.local_response_normalization(convnet)
	convnet = Activation('relu')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Conv2D(120, 5, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = tf.nn.local_response_normalization(convnet)
	convnet = Activation('relu')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Conv2D(160, 3, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	
	convnet = Conv2D(200, 3, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	flatten=Flatten()(convnet)
	convnet = Dense(320, activation='relu')(convnet)
	convnet = Dropout(dropout_prob)(convnet)

	convnet = Dense(320, activation='relu')(convnet)
	convnet = Dropout(dropout_prob)(convnet)

	preds = Dense(n_classes, activation='softmax')(convnet)

	model=Model(input=[input_img],output=[prediction])

	# checkpointer = ModelCheckpoint(filepath='./saved_models/weights/patch_based_cnn_101.hdf5', monitor='val_acc',verbose=1, save_best_only=True)
	# rmsprop=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
	sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=True)
	# adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])

	return model

def train_and_validate(model, training_generator, validation_generator, n_epochs=2):
	model.fit_generator(
    training_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=n_epochs,
    verbose=1,
    validation_data=validation_generator,
	validation_steps=val_samples// batch_size)
	# callbacks=[checkpointer,csv_logger])

##################################################################
#--------------------- EM Algo Helper functions -------------------#
##################################################################
def generate_predicted_maps(model, train_data_path):
	train_img_wise_patches_path = os.path.join(img_wise_patches_path, "train")
	img_dirname_list = os.listdir(train_img_wise_patches_path)
	label_0_file = os.path.join(img_wise_patches_path, "class_wise_probab_maps", "label_0.npy")
	label_1_file = os.path.join(img_wise_patches_path, "class_wise_probab_maps", "label_1.npy")
	
	label_probab = [1, 2]

	for img_dirname in img_dirname_list:
		patches_dir = os.path.join(train_img_wise_patches_path, img_dirname)
		patches_file = os.path.join(patches_dir, "patches.npy")
		probab_file = os.path.join(patches_dir, "probability_map.npy")
		patches = np.load(patches_file)
		img_label = int(img_dirname.split("_")[-1])

		predicted_map_array = np.zeros((1, 2))
		predicted_map_array = np.delete(predicted_map_array, [0], axis=0)

		while True:
			if patches.shape[0] >= max_patches_per_prediction:
				prediction = sess.run(preds, feed_dict={	img: patches[0:max_patches_per_prediction],
																	K.learning_phase(): 0})
				predicted_map_array = np.concatenate((predicted_map_array, prediction))
				patches = np.delete(patches, list(range(max_patches_per_prediction)), axis=0)

			else:
				prediction = sess.run(preds, feed_dict={	img: patches,
															K.learning_phase(): 0})
				predicted_map_array = np.concatenate((predicted_map_array, prediction))
				break

		# Probability map corresponding to the bag's label
		predicted_map_array = predicted_map_array[:, img_label]
		
		if type(label_probab[img_label]) is np.ndarray:
			label_probab[img_label] = np.concatenate((label_probab[img_label], predicted_map_array))
		else:
			label_probab[img_label] = predicted_map_array

		np.save(probab_file, predicted_map_array)

		print("Generated maps for : " + img_dirname + " " + str(img_dirname_list.index(img_dirname)+1))

	np.save(label_1_file, label_probab[1])												# Saves the probab_maps classwise for further processing
	np.save(label_0_file, label_probab[0])


def E_step(img_wise_patches, data_path, batch_size=128, img_lvl_pctl=30, class_lvl_pctl=30, n_classes=2):
	""" - Applies gaussian smoothing to the predicted probability maps
		- Generates bit mask for acceptable()discriminative patches for the next M_step
		- Returns:
				Refined training data for the next step
	"""
	delete_mask = list(range(batch_size))

	train_img_wise_patches_path = os.path.join(img_wise_patches, "train")
	train_data_path = os.path.join(data_path, "train")
	img_dirname_list = os.listdir(train_img_wise_patches_path)
	
	# Only for dir that have are meant to be updated regularly
	remove_and_create_dir(train_data_path)

	# For class lvl thresh
	label_0_file = os.path.join(img_wise_patches_path, "class_wise_probab_maps", "label_0.npy")
	label_1_file = os.path.join(img_wise_patches_path, "class_wise_probab_maps", "label_1.npy")
	label_0_probab_map = np.load(label_0_file)
	label_1_probab_map = np.load(label_1_file)
	class_lvl_thresh = [np.percentile(label_0_probab_map, class_lvl_pctl, axis=0), np.percentile(label_1_probab_map, class_lvl_pctl, axis=0)]

	patches = np.zeros((1,101,101,3))
	patches = np.delete(patches, [0], axis=0)

	labels = np.zeros((1,2))
	labels = np.delete(labels, [0], axis=0)

	batch_number = 1

	for img_dirname in img_dirname_list:										# iterating over images
		reconstructed_probab_map = np.zeros([2000,2000])
		
		source_patches_file = os.path.join(train_img_wise_patches_path, img_dirname, "patches.npy")
		source_label_file = os.path.join(train_img_wise_patches_path, img_dirname, "label.npy")
		source_coord_file = os.path.join(train_img_wise_patches_path, img_dirname, "coord.npy")
		source_probab_file = os.path.join(train_img_wise_patches_path, img_dirname, "probability_map.npy")

		source_patches = np.load(source_patches_file)
		source_labels = np.load(source_label_file)
		source_coords = np.load(source_coord_file)
		source_probab_maps = np.load(source_probab_file)
		print("E-step initiated for {}".format(img_dirname))
		print("Initial Shape of patches :", source_patches.shape)
		print("labels :", source_labels.shape)
		print("coords :", source_coords.shape)
		print("probab_maps :", source_probab_maps.shape)

		for patch_index in range(source_patches.shape[0]):										# Iterating over the patches
			center_coord = list(source_coords[patch_index])
			probability = source_probab_maps[patch_index]
			orig_patch_probab = reconstructed_probab_map[center_coord[0]-50: center_coord[0]+51, center_coord[1]-50:center_coord[1]+51]
			new_patch_prob = probability*GaussianKernel()
			# print(new_patch_prob.shape)
			# print(orig_patch_probab.shape)
			reconstructed_probab_map[center_coord[0]-50: center_coord[0]+51, center_coord[1]-50:center_coord[1]+51] = \
					np.maximum(orig_patch_probab, new_patch_prob)
		print("Reconstruction done successfully!!")

		img_lvl_thresh = np.percentile(source_probab_maps, img_lvl_pctl, axis=0)
		print("class_lvl_thresh :", class_lvl_thresh)
		print("img_lvl_thresh :", img_lvl_thresh)
		# print(labels.shape)
		threshold = min(img_lvl_thresh, class_lvl_thresh[np.argmax(source_labels[0,:])])
		discriminative_mask = (reconstructed_probab_map>threshold) | (reconstructed_probab_map==threshold)
		

		for patch_index in range(source_patches.shape[0]):
			coord = source_coords[patch_index]
			if discriminative_mask[coord[0], coord[1]]:
				patch = np.reshape(source_patches[patch_index], [1, 101, 101, 3])
				label = np.reshape(source_labels[patch_index], [1, 2])

				patches = np.concatenate((patches, patch))
				labels = np.concatenate((labels, label))
		
		print("Final shape of patches :", patches.shape)
		print("Updating training_data in batches now")

		while True:
			batch_dir = os.path.join(train_data_path, "batch_{}".format(batch_number))
			dest_patches_file = os.path.join(batch_dir, "patches.npy")
			dest_label_file = os.path.join(batch_dir, "label.npy")
			# print(patches.shape)

			if patches.shape[0] >= batch_size:
				if not os.path.exists(batch_dir):
					os.mkdir(batch_dir)

				np.save(dest_patches_file, patches[0:batch_size])
				np.save(dest_label_file, labels[0:batch_size])

				patches = np.delete(patches, delete_mask, axis=0)
				labels = np.delete(labels, delete_mask, axis=0)
			else:
				if img_dirname_list[-1] is img_dirname:
					if not os.path.exists(batch_dir):
						os.mkdir(batch_dir)

					np.save(dest_patches_file, patches)
					np.save(dest_label_file, labels)						
				break

			batch_number += 1


#################################################################
# -------------------- Other Helper functions ------------------#
#################################################################
def load_patch(img_path):
	return cv2.imread(img_path)

def GaussianKernel(ksize=101, nsig=30):
	gauss1D = cv2.getGaussianKernel(ksize, nsig)
	gauss2D = gauss1D*np.transpose(gauss1D)
	gauss2D = gauss2D/gauss2D[int(ksize/2), int(ksize/2)]
	return gauss2D

if __name__ == '__main__':
	main()


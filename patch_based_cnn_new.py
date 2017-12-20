from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import scipy.stats as st
import tensorflow as tf
import cv2
import shutil
import time

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam,SGD, RMSprop
from keras import backend as K


# --------------------- Global Variables -------------------------
# model_path = './saved_models'
# patient_wise_patches_path = './anno_cent'
# data_path = './data_dump'
# tmp_train_data_path = './train_tmp'									# tmp dir for training. will be deleted b4 the program closes
# val_data_path = data_path + '/valid'
# tmp_probab_path = './probab_maps_dump'

model_path = './saved_models'
patient_wise_patches_path = './tmp/anno_cent'
data_path = './tmp/data_dump'
tmp_train_data_path = './tmp/train_tmp'									# tmp dir for training. will be deleted b4 the program closes
val_data_path = data_path + '/valid'
tmp_probab_path = './tmp/probab_maps_dump_tmp'


n_iter = 20
batch_size = 128
n_classes = 2
data_augmentation = True

# input image dimensions
img_rows, img_cols = 101,101
img_channels = 3

# @todo : Description


# --------------------- Main Function ----------------------------
def main():
	# Default Session for Keras
	sess = tf.Session()
	K.set_session(sess)

	# Maintains a tmp directory for training, due to regular updations made on the directory
	if os.path.exists(tmp_train_data_path):
		shutil.rmtree(tmp_train_data_path)

	shutil.copytree(os.path.join(data_path, "train"), tmp_train_data_path)
	

	# Defining a model
	model = patch_based_cnn_model()

	# This will do preprocessing and realtime data augmentation:
	training_generator, validation_generator = data_generator(tmp_train_data_path, \
		val_data_path, batch_size=batch_size)

	#################### Initial M-step ######################## 
	# Training and predictions of probability maps
	train_and_validate(model, training_generator, validation_generator, tmp_train_data_path, val_data_path,\
						batch_size=batch_size)														#TODO

	print("First iteration of EM algo over")

	if n_iter > 1:
		print("LOADING INDICES NOW ........")
		start_time = time.time()
		img_wise_indices, patch_wise_indices = load_indices(tmp_train_data_path)
		print("--- %s seconds ---" % (time.time() - start_time))

	#################### 2nd Iteration Onwards ########################
	for itr in range(n_iter-1):
		if os.path.exists(tmp_probab_path):
			shutil.rmtree(tmp_probab_path)
		os.mkdir(tmp_probab_path)
		for label in range(n_classes):
			os.mkdir(os.path.join(tmp_probab_path, "label_"+str(label)))

		# E-step
		generate_predicted_maps(model, tmp_train_data_path, tmp_probab_path, img_wise_indices)		#TODO
		print("Probab maps predicted!!")
		# raw_input('Halt')

		E_step(tmp_train_data_path, tmp_probab_path, img_wise_indices, patch_wise_indices)			#TODO	
		img_wise_indices, patch_wise_indices = load_indices(tmp_train_data_path)
		print("{}th iteration's E-step performed!!".format(itr+1))
		# raw_input('Halt')		
		shutil.rmtree(tmp_probab_path)

		# M-Step
		train_and_validate(model, training_generator, validation_generator, tmp_train_data_path, val_data_path,\
						batch_size=batch_size)														#TODO
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
	# convnet = tf.nn.local_response_normalization(convnet)								#FIXME
	convnet = Activation('relu')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Conv2D(120, 5, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	# convnet = tf.nn.local_response_normalization(convnet)								#FIXME
	convnet = Activation('relu')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Conv2D(160, 3, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	
	convnet = Conv2D(200, 3, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Flatten()(convnet)
	convnet = Dense(320, activation='relu')(convnet)
	convnet = Dropout(dropout_prob)(convnet)

	convnet = Dense(320, activation='relu')(convnet)
	convnet = Dropout(dropout_prob)(convnet)

	preds = Dense(n_classes, activation='softmax')(convnet)

	model=Model(input=[input_img],output=[preds])

	# checkpointer = ModelCheckpoint(filepath='./saved_models/weights/patch_based_cnn_101.hdf5', monitor='val_acc',verbose=1, save_best_only=True)
	# rmsprop=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
	sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=True)
	# adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])

	return model

def data_generator(train_data_path, val_data_path, batch_size=128, img_rows=101, img_cols=101):
	datagen = ImageDataGenerator(rescale=1.0/255.)
	datagen_augmented = ImageDataGenerator(	featurewise_center=False,
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
	    train_data_path,
	    target_size=(img_cols, img_rows),
	    batch_size=batch_size,
	    class_mode='categorical')
	validation_generator = datagen.flow_from_directory(
	    val_data_path,
	    target_size=(img_cols, img_rows),
	    batch_size=batch_size,
	    class_mode='categorical')
	return [train_generator_augmented, validation_generator]

def train_and_validate(model, training_generator, validation_generator, train_data_path, val_data_path, n_epochs=2, batch_size=128):
	
	train_samples=len(os.listdir(tmp_train_data_path+"/label_0/"))+len(os.listdir(tmp_train_data_path+"/label_1/"))
	val_samples=len(os.listdir(val_data_path+"/label_0/"))+len(os.listdir(val_data_path+"/label_1/"))
	
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
def load_indices(train_data_path, n_classes=2):
 	img_wise_indices = []
 	patch_wise_indices = []

 	for label in range(n_classes):
 		label_img_wise_indices = {}
 		label_patch_wise_indices = {}

 		patch_list = os.listdir(train_data_path+"/label_"+str(label))

 		for patch_index in range(len(patch_list)):
 			patch_name = patch_list[patch_index].split(".")[0]
 			patch_split = patch_name.split("_")
 			img_name = "_".join([patch_split[0],]+patch_split[3:])

 			if img_name not in label_img_wise_indices.keys():
 				label_img_wise_indices[img_name] = [patch_index,]
 			else:
 				label_img_wise_indices[img_name] += [patch_index,]

 			label_patch_wise_indices[patch_name]={}
 			label_patch_wise_indices[patch_name]["index"]=patch_index
 			label_patch_wise_indices[patch_name][img_name]=img_name
 			label_patch_wise_indices[patch_name]["coord"]=[int(patch_split[1]), int(patch_split[2])]
	 		
		img_wise_indices += [label_img_wise_indices,]
		patch_wise_indices += [label_patch_wise_indices,]

	print("Length of patch wise indices for label_0:", len(patch_wise_indices[0]),"!!!!")
	print("Length of patch wise indices for label_1:", len(patch_wise_indices[1]),"!!!!")
	return img_wise_indices, patch_wise_indices


def generate_predicted_maps(model, train_data_path, probab_path, img_wise_indices, n_classes=2):

	for label in range(n_classes):
		class_data_path = os.path.join(train_data_path, "label_"+str(label))
		patch_list = os.listdir(class_data_path)

		class_probab_map = np.zeros((1, 2))
		class_probab_map = np.delete(class_probab_map, [0], axis=0)

		for img_name in img_wise_indices[label].keys():
			patches = np.zeros((1, 101, 101, 3))
			patches = np.delete(patches, [0], axis=0)
			
			for patch_index in img_wise_indices[label][img_name]:
				patch_path = os.path.join(class_data_path, patch_list[patch_index])
				patch = load_patch(patch_path)
				patches = np.concatenate((patches, np.expand_dims(patch, axis=0)))
			
			img_probab_map = model.predict(patches, batch_size=1024)
			np.save(os.path.join(probab_path, "label_"+str(label), img_name+".npy"), img_probab_map[:,label])

			class_probab_map = np.concatenate((class_probab_map, img_probab_map))
			print("Predicted map completed for", img_name)

		np.save(os.path.join(probab_path, "label_"+str(label)+".npy"), class_probab_map)
	print("PREDICTED ALL THE MAPS")

def E_step(train_data_path, probab_path, img_wise_indices, patch_wise_indices, img_lvl_pctl=10, class_lvl_pctl=10, n_classes=2):

	for label in range(n_classes):
		class_probab_map = np.load(os.path.join(probab_path, "label_"+str(label)+".npy"))
		class_lvl_thresh = np.percentile(class_probab_map, class_lvl_pctl)

		class_data_path = os.path.join(train_data_path, "label_"+str(label))
		patch_list = os.listdir(class_data_path)

		for img_name in img_wise_indices[label].keys():
			img_probab_map = np.load(os.path.join(probab_path, "label_"+str(label), img_name+".npy"))
			img_lvl_thresh = np.percentile(img_probab_map, img_lvl_pctl)
			print("image probab_map shape:", img_probab_map.shape)
			print("class probab map shape:", class_probab_map.shape)

			reconstructed_probab_map = np.zeros([2000,2000])

			for index in range(img_probab_map.shape[0]):
				patch_index = img_wise_indices[label][img_name][index]
				patch_name = patch_list[patch_index].split(".")[0]
				patch_cent_coord = patch_wise_indices[label][patch_name]["coord"]

				probability = img_probab_map[index]
				orig_patch_probab = reconstructed_probab_map[patch_cent_coord[0]-50: patch_cent_coord[0]+51, \
										patch_cent_coord[1]-50:patch_cent_coord[1]+51]
				new_patch_prob = probability*GaussianKernel()
				reconstructed_probab_map[patch_cent_coord[0]-50: patch_cent_coord[0]+51, patch_cent_coord[1]-50:patch_cent_coord[1]+51] = \
					np.maximum(orig_patch_probab, new_patch_prob)
			print("IMAGE :", img_name)
			# print("Reconstruction done successfully!!")
			print("class_lvl_thresh :", class_lvl_thresh)
			print("img_lvl_thresh :", img_lvl_thresh)

			threshold = min(img_lvl_thresh, class_lvl_thresh)
			discriminative_mask = (reconstructed_probab_map>threshold) | (reconstructed_probab_map==threshold)

			for patch_name in patch_wise_indices[label].keys():
				patch_cent_coord = patch_wise_indices[label][patch_name]["coord"]
				if discriminative_mask[patch_cent_coord[0], patch_cent_coord[1]] is False:
					os.remove(os.path.join(train_data_path, "label_"+str(label), patch_name+".png"))


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
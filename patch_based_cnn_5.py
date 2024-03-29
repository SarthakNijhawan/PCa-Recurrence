# from __future__ import print_function
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
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Input, Flatten, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam,SGD, RMSprop
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

# --------------------- Global Variables -------------------------
data_path = './expt_5/data_dump'
val_data_path = 'expt_5/data_dump/valid'
discarded_patches_path = './expt_5/discarded_patches'
reconstructed_probab_map_path = './expt_5/reconstructed_maps'
tmp_train_data_path = './expt_5/useful_patches'									# tmp dir for training. will be deleted b4 the program closes
tmp_probab_path = './expt_5/probab_maps_dump_tmp'
model_weight_path = './expt_5/expt_5_cnn_weights.h5'
csv_logger_path = 'expt_5/expt_5_cnn.csv'

# Model hyper-paramenters
n_iter=10
batch_size=32
n_classes=2
data_augmentation=True
img_lvl_pctl=10
class_lvl_pctl=20
dropout_prob=0.0
l_rate=0.001
discr_map_size=[31, 31]
strides=50
coord_offset=150

# input image dimensions
img_rows, img_cols=100,100
img_channels=3

""" Features of this exp
		- Slight modifications made in the architecture
		- Patches from outside the annotation area also includes
		- Patches extracted in continuous strides rather than centric to a nucleus
		- Only Gaussian smoothing needed now, no req of a gaussian kernel for reconstruction of the whole image
		- Ony a patch based CNN is trained (i.e. First lvl of training)
"""

# --------------------- Main Function ----------------------------
def main():
	# sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	sess=tf.Session()
	K.set_session(sess)
	
	# Maintains a tmp directory for training, due to regular updations made on the directory
	# if os.path.exists(tmp_train_data_path):
	# 	shutil.rmtree(tmp_train_data_path)

	# print("Making a tmp storage for train data.........")
	# shutil.copytree(os.path.join(data_path, "train"), tmp_train_data_path)
	
	# discarded patches dir
	make_dir(discarded_patches_path, False)
	make_dir(reconstructed_probab_map_path)

	# Defining a model
	model = patch_based_cnn_model()
	print model.summary()
	# This will do preprocessing and realtime data augmentation:
	train_data_gen, data_gen = data_generator()

	#################### Initial M-step ######################## 
	# img_wise_indices, patch_wise_indices = load_indices(tmp_train_data_path)
	# model.load_weights(model_weight_path)
	# E_step(img_wise_indices, patch_wise_indices, 2)
	# shutil.rmtree(tmp_probab_path)
	# print("{} iteration ::::::::::: E-step performed!!".format(itr+2))
	
	######### M-Step #########
	print(":::::::::::::::::::::: {}th iteration ::::::::::::::::::::".format(1))
	train_and_validate(model, train_data_gen, data_gen)
	# print("{} iteration ::::::::::: M-step performed!!".format(itr+2))

	#################### 2nd Iteration Onwards ########################
	for itr in range(n_iter-1):
		make_dir(tmp_probab_path)
		print(":::::::::::::::::::::: {}th iteration ::::::::::::::::::::".format(itr+2))

		img_wise_indices, patch_wise_indices = load_indices(tmp_train_data_path)
		print("Length of patch wise indices for label_0:", len(patch_wise_indices[0]))
		print("Length of patch wise indices for label_1:", len(patch_wise_indices[1]))

		######### E-step #########
		generate_predicted_maps(model, data_gen, img_wise_indices)
		# print("................... Probability maps predicted .............")
		E_step(img_wise_indices, patch_wise_indices, itr+2)
		shutil.rmtree(tmp_probab_path)
		# print("{} iteration ::::::::::: E-step performed!!".format(itr+2))
		
		######### M-Step #########
		train_and_validate(model, train_data_gen, data_gen)
		# print("{} iteration ::::::::::: M-step performed!!".format(itr+2))

	# EM-Algo completed
	sess.close()
	print("Training Completed Successfully !!")


##################################################################
#---------------------Model and its functions -------------------#
##################################################################
def patch_based_cnn_model():
	# Layers
	input_img=Input(shape=(img_rows,img_cols,3))
	convnet = BatchNormalization()(input_img)

	convnet = Conv2D(80, 5, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = BatchNormalization(axis=3)(convnet)
	convnet = Activation('relu')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Conv2D(120, 5, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = BatchNormalization(axis=3)(convnet)
	convnet = Activation('relu')(convnet)
	
	convnet = Conv2D(160, 3, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = BatchNormalization(axis=3)(convnet)
	convnet = Activation('relu')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Conv2D(200, 3, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = BatchNormalization(axis=3)(convnet)
	convnet = Activation('relu')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Flatten()(convnet)
	convnet = Dense(320, activation='relu')(convnet)
	# convnet = Dropout(dropout_prob)(convnet)

	convnet = Dense(320, activation='relu')(convnet)
	# convnet = Dropout(dropout_prob)(convnet)

	preds = Dense(n_classes, activation='softmax')(convnet)
	model=Model(input=[input_img],output=[preds])

	sgd = SGD(lr=l_rate, momentum=0.0, decay=0.0, nesterov=True)
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])
	return model

def data_generator():
	datagen = ImageDataGenerator(rescale=1.0/255.)
	datagen_augmented = ImageDataGenerator(	featurewise_center=False,
	    samplewise_center=False,
	    featurewise_std_normalization=False,
	    samplewise_std_normalization=False,
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
	return datagen_augmented, datagen

def train_and_validate(model, train_data_gen, valid_data_gen, n_epochs=2):
	train_samples_0 = len(os.listdir(tmp_train_data_path+"/label_0/"))
	train_samples_1 = len(os.listdir(tmp_train_data_path+"/label_1/"))  
	train_samples = train_samples_0 + train_samples_1
	
	val_samples=len(os.listdir(val_data_path+"/label_0/"))+len(os.listdir(val_data_path+"/label_1/"))
	
	train_generator_augmented = train_data_gen.flow_from_directory(
	    tmp_train_data_path,
	    target_size=(img_cols, img_rows),
	    batch_size=batch_size,
	    class_mode='categorical')

	validation_generator = valid_data_gen.flow_from_directory(
	    val_data_path,
	    target_size=(img_cols, img_rows),
	    batch_size=batch_size,
	    class_mode='categorical')

	checkpointer = ModelCheckpoint(filepath=model_weight_path, monitor='val_acc',verbose=1, save_best_only=True)
	csv_logger = CSVLogger(csv_logger_path)
	class_weight = {0 : 1.0*train_samples_1/train_samples_0 if train_samples_0 < train_samples_1 else 1.0,
				    1 : 1.0*train_samples_0/train_samples_1 if train_samples_1 < train_samples_0 else 1.0}
	model.fit_generator(
		    train_generator_augmented,
		    steps_per_epoch=train_samples // batch_size,
		    epochs=n_epochs,
		    verbose=1,
		    validation_data=validation_generator,
			validation_steps=val_samples // batch_size,
			callbacks=[checkpointer, csv_logger],
			class_weight=class_weight)

##################################################################
#--------------------- EM Algo Helper functions -------------------#
##################################################################
def load_indices(data_path):
 	img_wise_indices = []
 	patch_wise_indices = []

 	for label in range(n_classes):
 		label_img_wise_indices = {}
 		label_patch_wise_indices = {}

 		patch_list = os.listdir(data_path+"/label_"+str(label))
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
 			label_patch_wise_indices[patch_name]["img_name"]=img_name
 			label_patch_wise_indices[patch_name]["coord"]=[int(patch_split[1]), int(patch_split[2])]
	 		
		img_wise_indices += [label_img_wise_indices,]
		patch_wise_indices += [label_patch_wise_indices,]

	return img_wise_indices, patch_wise_indices

def generate_predicted_maps(model, test_datagen, img_wise_indices):

	for label in range(n_classes):
		class_data_path = os.path.join(tmp_train_data_path, "label_"+str(label))
		patch_list = os.listdir(class_data_path)

		class_probab_map = np.zeros((1,))
		class_probab_map = np.delete(class_probab_map, [0], axis=0)

		# Iterating img_wise over the patches
		for img_name in img_wise_indices[label].keys():
			patches = np.zeros((1, img_rows, img_cols, 3))
			patches = np.delete(patches, [0], axis=0)
			
			# For all the patches of the image
			for patch_index in img_wise_indices[label][img_name]:
				patch_path = os.path.join(class_data_path, patch_list[patch_index])
				patch = load_patch(patch_path)
				patches = np.concatenate((patches, np.expand_dims(patch, axis=0)))
			
			model.load_weights(model_weight_path)
			test_samples = patches.shape[0]
			test_generator = test_datagen.flow(
		        patches,
		        shuffle = False,
		        batch_size = batch_size)
			img_probab_map = model.predict_generator(test_generator, steps=test_samples // batch_size)
			img_probab_map = img_probab_map[:, label]

			# saves the probab_map for the img
			np.save(os.path.join(tmp_probab_path, "label_"+str(label), img_name+".npy"), img_probab_map)
			class_probab_map = np.concatenate((class_probab_map, img_probab_map))

		# Saves the class lvl probab maps
		np.save(os.path.join(tmp_probab_path, "label_"+str(label)+".npy"), class_probab_map)

def E_step(img_wise_indices, patch_wise_indices, iteration):

	for label in range(n_classes):
		class_probab_map = np.load(os.path.join(tmp_probab_path, "label_"+str(label)+".npy"))
		class_lvl_thresh = np.percentile(class_probab_map, class_lvl_pctl)

		class_data_path = os.path.join(tmp_train_data_path, "label_"+str(label))
		patch_list = os.listdir(class_data_path)

		for img_name in img_wise_indices[label].keys():
			img_probab_map = np.load(os.path.join(tmp_probab_path, "label_"+str(label), img_name+".npy"))
			img_lvl_thresh = np.percentile(img_probab_map, img_lvl_pctl)

			# This percentile will help form the base for the probability map
			black_region_val = np.percentile(img_probab_map, class_lvl_pctl)

			# Constructing the whole image lvl probability map from the patch probabilities
			reconstructed_probab_map = np.ones(discr_map_size)*black_region_val
			
			# Iterating over all the patches of the image
			for index in range(img_probab_map.shape[0]):
				patch_index = img_wise_indices[label][img_name][index]
				patch_name = patch_list[patch_index].split(".")[0]
				patch_cent_coord = patch_wise_indices[label][patch_name]["coord"]
				x_coord = int((1.0*patch_cent_coord[0]-img_rows/2-coord_offset)/strides)
				y_coord = int((1.0*patch_cent_coord[1]-img_cols/2-coord_offset)/strides)
				probability = img_probab_map[index]
				reconstructed_probab_map[x_coord, y_coord] = probability

			# Gaussian Smoothing on the reconstructed image
			gauss_map = cv2.GaussianBlur(np.uint8(reconstructed_probab_map*255), (3,3), 0)
			
			threshold = min(img_lvl_thresh, class_lvl_thresh)
			discriminative_mask = gauss_map >= threshold*255

			# Saving visualisable probab and discriminative maps
			img_recons_path = os.path.join(reconstructed_probab_map_path, "label_"+str(label), img_name)
			if not os.path.exists(img_recons_path):
				os.mkdir(img_recons_path)
			img_gauss_file = os.path.join(img_recons_path, str(iteration)+"_gauss.png")
			img_discrim_file = os.path.join(img_recons_path, str(iteration)+"_discriminative.png")
			img_recons_file = os.path.join(img_recons_path, str(iteration)+"_reconstructed.png")

			cv2.imwrite(img_gauss_file, gauss_map)
			cv2.imwrite(img_recons_file, np.uint8(reconstructed_probab_map*255))
			cv2.imwrite(img_discrim_file, np.uint8(255*(1*discriminative_mask)))
			
			for index in range(img_probab_map.shape[0]):
				patch_index = img_wise_indices[label][img_name][index]
				patch_name = patch_list[patch_index].split(".")[0]
				patch_cent_coord = patch_wise_indices[label][patch_name]["coord"]
				x_coord = int((1.0*patch_cent_coord[0]-img_rows/2-coord_offset)/strides)
				y_coord = int((1.0*patch_cent_coord[1]-img_cols/2-coord_offset)/strides)
				if discriminative_mask[x_coord, y_coord] == False:
					shutil.move(os.path.join(tmp_train_data_path, "label_"+str(label), patch_name+".png"), \
						discarded_patches_path)

#################################################################
# -------------------- Other Helper functions ------------------#
#################################################################
def load_patch(img_path):
	return cv2.imread(img_path)

def GaussianKernel(ksize=101, nsig=25):
	gauss1D = cv2.getGaussianKernel(ksize, nsig)
	gauss2D = gauss1D*np.transpose(gauss1D)
	gauss2D = gauss2D/gauss2D[int(ksize/2), int(ksize/2)]
	return gauss2D

def make_dir(dir, needlabel=True):
	if os.path.exists(dir):
		shutil.rmtree(dir)
	os.mkdir(dir)
	if needlabel is True:
		for label in range(n_classes):
			os.mkdir(os.path.join(dir, "label_"+str(label)))

if __name__ == '__main__':
	main()
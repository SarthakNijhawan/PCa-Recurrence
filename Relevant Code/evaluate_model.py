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


# --------------------- Global Variables -------------------------
data_path = 'data_dump'
val_data_path = 'data_dump/test_npy'
model_weight_path = 'expt_5_cnn_weights.h5'
csv_logger_path = 'expt_5_cnn.csv'

# Model hyper-paramenters
val_thresh = 0.7
batch_size=32
n_classes=2
l_rate=0.001
data_augmentation=True

# input image dimensions
img_rows, img_cols=100,100
img_channels=3

""" Description:
		- Evaluates the 2-lvl MIL trained model
		- Here the 2nd lvl model is assumed to be count based voting
"""

# --------------------- Main Function ----------------------------
def main():
	sess=tf.Session()
	K.set_session(sess)
	
	# Defining a model
	model = patch_based_cnn_model()
	print model.summary()
	
	train_data_gen, val_data_gen = data_generator()
	# patch_wise_indices, img_wise_indices = load_indices(val_data_path)

	evaluate_model(model, val_data_gen)

	sess.close()

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

def evaluate_model(model, valid_data_gen):

	predictions = []

	for label in range(n_classes):
		val_patches_path = os.path.join(val_data_path, "label_"+str(label))
		img_wise_patches_list = os.listdir(val_patches_path)
		
		for img_wise_patches in img_wise_patches_list:
			patches_path = os.path.join(val_patches_path, img_wise_patches)
			patches = np.load(patches_path)

			model.load_weights(model_weight_path)
			valid_samples = patches.shape[0]
			val_generator = valid_data_gen.flow(
						        patches,
						        shuffle = False,
						        batch_size = batch_size)
			patches_pred = model.predict_generator(val_generator, steps=valid_samples // batch_size)

			# patches_pred = model.predict(patches)
			patches_labels = np.argmax(patches_pred, axis=1)

			discr_map = np.logical_xor(patches_pred>val_thresh, patches_pred<(1-val_thresh))
			discr_map = np.logical_and(discr_map[:,0], discr_map[:,1])

			patches_labels = patches_labels[discr_map]
			same_label_patches = len(patches_labels[patches_labels==label])
			diff_label_patches = len(patches_labels[patches_labels==(1-label)])
			img_lvl_pred = 1*(same_label_patches > diff_label_patches)
			print(img_wise_patches.split(".")[0], ":", img_lvl_pred, \
				 "and patches :", valid_samples, "same_label :", same_label_patches, \
				 "diff_label :", diff_label_patches)
			predictions += [img_lvl_pred,]

	predictions = np.array(predictions)
	accuracy = np.mean(predictions)
	print("Accuracy of the model:", accuracy)

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

if __name__ == '__main__':
	main()

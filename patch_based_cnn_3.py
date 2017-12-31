# from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
model_path = './expt_3/saved_models'
data_path = './expt_3/data_dump'
tmp_train_data_path = './expt_3/useful_patches'									# tmp dir for training. will be deleted b4 the program closes
val_data_path = data_path + '/valid'
tmp_probab_path = './expt_3/probab_maps_dump_tmp'
discarded_patches = './expt_3/discarded_patches'
reconstructed_probab_map_path = './expt_3/reconstructed_maps'

model_weight_path = './expt_3/expt_3_cnn_weights.h5'

# Model hyper-paramenters
n_iter=10
batch_size=32
n_classes=2
data_augmentation=True
img_lvl_pctl=5
class_lvl_pctl=10
dropout_prob=0.0
l_rate=0.0001

# input image dimensions
img_rows, img_cols=101,101
img_channels=3

""" Features of this exp
		- Img_lvl_decision_fusion model (Voting)
		- Same arch as that of paper
		- Additional Gaussian Smoothing
		- batch = 256
		- img_lvl_pctl = 5
		- class_lvl_pctl = 7
		- starting epoch = 2
		- loss = binary crossentropy
		- nsig = 20
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
	make_dir(discarded_patches, False)
	make_dir(reconstructed_probab_map_path)

	# Defining a model
	model = patch_based_cnn_model()
	print model.summary()
	# This will do preprocessing and realtime data augmentation:
	train_data_gen, data_gen = data_generator()

	#################### Initial M-step ######################## 
	# Training and predictions of probability maps
	train_and_validate(model, train_data_gen, data_gen, tmp_train_data_path, val_data_path)
	print("First iteration of EM algo over")

	#################### 2nd Iteration Onwards ########################
	for itr in range(n_iter-1):
		make_dir(tmp_probab_path)
		print(":::::::::::::::::::::: {}th iteration ::::::::::::::::::::".format(itr+2))

		img_wise_indices, patch_wise_indices = load_indices(tmp_train_data_path)
		print("Length of patch wise indices for label_0:", len(patch_wise_indices[0]))
		print("Length of patch wise indices for label_1:", len(patch_wise_indices[1]))

		######### E-step #########
		generate_predicted_maps(model, data_gen, tmp_train_data_path, tmp_probab_path, img_wise_indices)
		print("................... Probability maps predicted .............")
		E_step(tmp_train_data_path, tmp_probab_path, discarded_patches, img_wise_indices, patch_wise_indices, reconstructed_probab_map_path, itr+2)
		shutil.rmtree(tmp_probab_path)
		print("{} iteration ::::::::::: E-step performed!!".format(itr+2))
		
		######### M-Step #########
		train_and_validate(model, train_data_gen, data_gen, tmp_train_data_path, val_data_path)
		print("{} iteration ::::::::::: M-step performed!!".format(itr+2))

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

	convnet = Conv2D(32, 7, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = BatchNormalization(axis=3)(convnet)
	convnet = Activation('relu')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Conv2D(64, 3, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = BatchNormalization(axis=3)(convnet)
	convnet = Activation('relu')(convnet)
	
	convnet = Conv2D(128, 3, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(convnet)
	convnet = BatchNormalization(axis=3)(convnet)
	convnet = Activation('relu')(convnet)
	convnet = MaxPooling2D(pool_size=(2, 2), strides=2)(convnet)

	convnet = Flatten()(convnet)
	convnet = Dense(320, activation='relu')(convnet)
	convnet = Dropout(dropout_prob)(convnet)

	convnet = Dense(320, activation='relu')(convnet)
	convnet = Dropout(dropout_prob)(convnet)

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

def train_and_validate(model, train_data_gen, valid_data_gen, train_data_path, val_data_path, n_epochs=2):
	train_samples_0 = len(os.listdir(tmp_train_data_path+"/label_0/"))
	train_samples_1 = len(os.listdir(tmp_train_data_path+"/label_1/"))  
	train_samples = train_samples_0 + train_samples_1
	
	val_orig_data_path = os.path.join(data_path, 'valid_orig')
	val_samples=len(os.listdir(val_orig_data_path+"/label_0/"))+len(os.listdir(val_orig_data_path+"/label_1/"))
	
	train_generator_augmented = train_data_gen.flow_from_directory(
	    train_data_path,
	    target_size=(img_cols, img_rows),
	    batch_size=batch_size,
	    class_mode='categorical')

	validation_generator = valid_data_gen.flow_from_directory(
	    val_orig_data_path,
	    target_size=(img_cols, img_rows),
	    batch_size=batch_size,
	    class_mode='categorical')

	checkpointer = ModelCheckpoint(filepath=model_weight_path, monitor='val_acc',verbose=1, save_best_only=True)
	csv_logger = CSVLogger('expt_3/expt_3_cnn.csv')
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

	# 2nd Level Fusion model (Validation Part)
	val_thresh = 0.6
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
				"with output Shape :", patches_pred.shape, "and patches :",\
				 valid_samples, "same_label_patches :", same_label_patches, \
				 "diff_label_patches :", diff_label_patches)
			predictions += [img_lvl_pred,]

	predictions = np.array(predictions)
	accuracy = np.mean(predictions)
	print("Accuracy of the model:", accuracy)

##################################################################
#--------------------- EM Algo Helper functions -------------------#
##################################################################
def load_indices(train_data_path):
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
 			label_patch_wise_indices[patch_name]["img_name"]=img_name
 			label_patch_wise_indices[patch_name]["coord"]=[int(patch_split[1]), int(patch_split[2])]
	 		
		img_wise_indices += [label_img_wise_indices,]
		patch_wise_indices += [label_patch_wise_indices,]

	return img_wise_indices, patch_wise_indices

def generate_predicted_maps(model, test_datagen, train_data_path, probab_path, img_wise_indices):

	for label in range(n_classes):
		class_data_path = os.path.join(train_data_path, "label_"+str(label))
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
			np.save(os.path.join(probab_path, "label_"+str(label), img_name+".npy"), img_probab_map)
			class_probab_map = np.concatenate((class_probab_map, img_probab_map))

		# Saves the class lvl probab maps
		np.save(os.path.join(probab_path, "label_"+str(label)+".npy"), class_probab_map)

def E_step(train_data_path, probab_path, discard_patches_dir, img_wise_indices, patch_wise_indices, reconstructed_probab_map_path, iteration, \
	img_size=[2000, 2000]):

	for label in range(n_classes):
		class_probab_map = np.load(os.path.join(probab_path, "label_"+str(label)+".npy"))
		class_lvl_thresh = np.percentile(class_probab_map, class_lvl_pctl)

		class_data_path = os.path.join(train_data_path, "label_"+str(label))
		patch_list = os.listdir(class_data_path)

		for img_name in img_wise_indices[label].keys():
			img_probab_map = np.load(os.path.join(probab_path, "label_"+str(label), img_name+".npy"))
			img_lvl_thresh = np.percentile(img_probab_map, img_lvl_pctl)

			reconstructed_probab_map = np.zeros(img_size)
			
			# Iterating over all the patches of the image
			for index in range(img_probab_map.shape[0]):
				patch_index = img_wise_indices[label][img_name][index]
				patch_name = patch_list[patch_index].split(".")[0]
				patch_cent_coord = patch_wise_indices[label][patch_name]["coord"]
				probability = img_probab_map[index]
				orig_patch_probab = reconstructed_probab_map[patch_cent_coord[0]-img_rows/2: patch_cent_coord[0]+img_rows/2+1, \
										patch_cent_coord[1]-img_cols/2:patch_cent_coord[1]+img_cols/2+1]
				new_patch_prob = probability*GaussianKernel(img_rows, 20)
				# new_patch_prob = probability*UniformKernel()
				reconstructed_probab_map[patch_cent_coord[0]-img_rows/2: patch_cent_coord[0]+img_rows/2+1, \
										patch_cent_coord[1]-img_cols/2:patch_cent_coord[1]+img_cols/2+1] = \
										(orig_patch_probab+new_patch_prob)/2.0
										# np.maximum(orig_patch_probab, new_patch_prob)
										

			# Gaussian Smoothing on the reconstructed image
			gauss_map = cv2.GaussianBlur(np.uint8(reconstructed_probab_map*255),(11,11),0)
			
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
				if discriminative_mask[patch_cent_coord[0], patch_cent_coord[1]] == False:
					shutil.move(os.path.join(train_data_path, "label_"+str(label), patch_name+".png"), \
						discard_patches_dir)

#################################################################
# -------------------- Other Helper functions ------------------#
#################################################################
def load_patch(img_path):
	return cv2.imread(img_path)

def GaussianKernel(ksize=101, nsig=20):
	gauss1D = cv2.getGaussianKernel(ksize, nsig)
	gauss2D = gauss1D*np.transpose(gauss1D)
	gauss2D = gauss2D/gauss2D[int(ksize/2), int(ksize/2)]
	return gauss2D

def UniformKernel(ksize=101, sq_len=10):
	sq_len_shape = 2*sq_len + 1
	
	uniform2D = np.zeros((ksize, ksize))
	uniform2D[ksize/2-sq_len:ksize/2+sq_len+1, ksize/2-sq_len:ksize/2+sq_len+1] = \
		np.ones((sq_len_shape, sq_len_shape))
	return uniform2D

def make_dir(dir, needlabel=True):
	if os.path.exists(dir):
		shutil.rmtree(dir)
	os.mkdir(dir)
	if needlabel is True:
		for label in range(n_classes):
			os.mkdir(os.path.join(dir, "label_"+str(label)))

if __name__ == '__main__':
	main()

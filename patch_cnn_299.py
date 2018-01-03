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


# --------------------- Global Variables -------------------------
data_path = 'DB_HnE_299_dense'
val_data_path = os.path.join(data_path, 'valid')
train_data_path = os.path.join(data_path, 'train')

model_weight_path = 'saved_models/patch_lvl_cnn_weights.h5'
csv_logger_path = 'saved_models/patch_lvl_cnn_weights.csv'

# Model hyper-paramenters
batch_size=32
n_classes=2
data_augmentation=True
dropout_prob=0.0
l_rate=0.001
n_epochs=50

# input image dimensions
img_rows, img_cols=299, 299
img_channels=3


""" Description:
		- Naive patch lvl CNN classifier
		- patch_size = 299

"""

# --------------------- Main Function ----------------------------
def main():
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
	# sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	sess=tf.Session()
	K.set_session(sess)
	
	# Defining a model
	model = patch_based_cnn_model()
	print model.summary()
	# This will do preprocessing and realtime data augmentation:
	train_data_gen, data_gen = data_generator()

	train_and_validate(model, train_data_gen, data_gen)

	# Training completed
	sess.close()
	print("Training Completed Successfully !!")


##################################################################
#---------------------Model and its functions -------------------#
##################################################################
def patch_based_cnn_model():

	inputs = Input(shape=(img_rows, img_cols, 3))
	conv1 = Conv2D(6, 6, padding = 'valid', kernel_initializer = 'he_normal')(inputs)
	print("conv1 shape:",conv1.shape)
	# conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	conv1 = Dropout(0)(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv1 = Conv2D(12, 5, padding = 'valid', kernel_initializer = 'he_normal')(pool1)
	print("conv1 shape:",conv1.shape)
	# conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	conv1 = Dropout(0)(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	print("pool1 shape:",pool1.shape)

	conv2 = Conv2D(16, 4, padding = 'valid', kernel_initializer = 'he_normal')(pool1)
	print("conv2 shape:",conv2.shape)
	# conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	conv2 = Dropout(0)(conv2)
	conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv2 = Conv2D(32, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv2)
	print("conv2 shape:",conv2.shape)
	# conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	conv2 = Dropout(0.6)(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	print("pool2 shape:",pool2.shape)
	conv3 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
	print("conv2 shape:",conv3.shape)
	# conv2 = BatchNormalization()(conv2)
	conv3 = Activation('relu')(conv3)
	conv3 = Dropout(0)(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	print("pool2 shape:",pool3.shape)
	#conv4 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
	#print("conv2 shape:",conv4.shape)
	# conv2 = BatchNormalization()(conv2)
	#conv4 = Activation('relu')(conv4)
	#conv4 = Dropout(0.6)(conv4)
	#pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	#print("pool2 shape:",pool4.shape)
	#conv5 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
	#print("conv2 shape:",conv5.shape)
	# conv2 = BatchNormalization()(conv2)
	#conv5 = Activation('relu')(conv5)
	#conv5 = Dropout(0.6)(conv5)
	#pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
	print("pool2 shape:",pool3.shape)
	flatten = Flatten()(pool3)
	#fc_new1= Dense(512,activation='relu')(flatten)
	#drop_new1 = Dropout(0.9)(fc_new1)
	fc_new2 =Dense(64,activation='relu')(flatten)
	drop_new2=Dropout(0.6)(fc_new2)    
	fc_final_new =Dense(2, activation='softmax')(flatten)    
	model = Model(inputs = inputs, outputs = fc_final_new)

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

def train_and_validate(model, train_data_gen, valid_data_gen):
	train_samples_0 = len(os.listdir(train_data_path+"/label_0/"))
	train_samples_1 = len(os.listdir(train_data_path+"/label_1/"))  
	train_samples = train_samples_0 + train_samples_1
	
	val_samples=len(os.listdir(val_data_path+"/label_0/"))+len(os.listdir(val_data_path+"/label_1/"))
	
	train_generator_augmented = train_data_gen.flow_from_directory(
	    train_data_path,
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

if __name__ == '__main__':
	main()
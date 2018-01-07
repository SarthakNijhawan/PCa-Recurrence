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
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger


# --------------------- Global Variables -------------------------
data_path = './'
val_data_path = data_path+'valid'
discarded_patches_path = data_path+'discarded_patches'
reconstructed_probab_map_path = data_path+'reconstructed_maps'
tmp_train_data_path = data_path+'useful_patches'                                 # tmp dir for training. will be deleted b4 the program closes
tmp_probab_path = data_path+'probab_maps_dump_tmp'
model_name = "gleason_grade_299_anno_cent_mil"
model_weight_path = model_name+'.h5'
csv_logger_path = model_name+'.csv'
model_plot = model_name+".png"

# Model hyper-paramenters
n_iter=20
batch_size=32
data_augmentation=True
img_lvl_pctl=5
class_lvl_pctl=10
l_rate=0.001
n_epochs=2

# discr_map_size=[31, 31]
# strides=50
# coord_offset=150

n_classes=2
labels_list=[3, 45]

# input image dimensions
img_rows, img_cols=299,299
img_channels=3

""" Features of this exp
        - Application of MIL on patch based cnn for gleason grade prediction problem
        - 
"""

# --------------------- Main Function ----------------------------
def main():
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
	# sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	sess=tf.Session()
	K.set_session(sess)

	# Maintains a tmp directory for training, due to regular updations made on the directory
	if os.path.exists(tmp_train_data_path):
	  shutil.rmtree(tmp_train_data_path)

	print("Making a tmp storage for train data.........")
	shutil.copytree(os.path.join(data_path, "train"), tmp_train_data_path)

	# discarded patches dir
	make_dir(discarded_patches_path, False)
	make_dir(reconstructed_probab_map_path)

	# Defining a model
	model = patch_based_cnn_model()
	print model.summary()

	# This will do preprocessing and realtime data augmentation:
	train_data_gen, data_gen = data_generator()

	val_samples_0 = len(os.path.join(val_data_path, "label_3"))
	val_samples_1 = len(os.path.join(val_data_path, "label_45"))
	chance = 1.0*val_samples_0/(val_samples_1 + val_samples_0)
	print "Chance :", chance

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
    inputs = Input((img_rows, img_cols, 3))
    
    conv1 = Conv2D(16, 6, padding = 'valid', kernel_initializer = 'he_normal', dilation_rate=7)(inputs)
    print("conv1 shape:", conv1.shape)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    # conv1 = Dropout(0)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv1 = Conv2D(16, 5, padding = 'valid', kernel_initializer = 'he_normal', dilation_rate=6)(pool1)
    print("conv1 shape:", conv1.shape)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    # conv1 = Dropout(0)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1 shape:",pool1.shape)

    conv2 = Conv2D(16, 4, padding = 'valid', kernel_initializer = 'he_normal', dilation_rate=4)(pool1)
    print("conv2 shape:", conv2.shape)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    # conv2 = Dropout(0)(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv2 = Conv2D(32, 3, padding = 'valid', kernel_initializer = 'he_normal', dilation_rate=2)(conv2)
    print("conv2 shape:", conv2.shape)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    # conv2 = Dropout(0)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2 shape:", pool2.shape)
    
    conv3 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print("conv2 shape:", conv3.shape)
    # conv2 = BatchNormalization()(conv2)
    conv3 = Activation('relu')(conv3)
    # conv3 = Dropout(0)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print("pool2 shape:",pool3.shape)
    
    conv4 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    print("conv2 shape:", conv4.shape)
    # conv2 = BatchNormalization()(conv2)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.8)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print("pool2 shape:", pool4.shape)
    
    conv5 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    print("conv2 shape:", conv5.shape)
    # conv2 = BatchNormalization()(conv2)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.8)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    print("pool2 shape:", pool5.shape)
    
    flatten = Flatten()(pool5)
    #fc_new1= Dense(512,activation='relu')(flatten)
    #drop_new1 = Dropout(0.9)(fc_new1)
    fc_new2 =Dense(64,activation='relu')(flatten)
    drop_new2=Dropout(0.8)(fc_new2)    
    
    fc_final_new =Dense(2,activation='softmax')(drop_new2)    
    model = Model(inputs = inputs, outputs = fc_final_new)

    # sgd = SGD(lr=l_rate, momentum=0.0, decay=0.0, nesterov=True)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])

    model.compile(optimizer = Adam(lr=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
    plot_model(model, to_file=model_plot,show_shapes=True)

    return model

def data_generator():
	datagen = ImageDataGenerator(rescale=1.0/255.)
	datagen_augmented = ImageDataGenerator(featurewise_center=False,
		samplewise_center=False,
		featurewise_std_normalization=False,
		samplewise_std_normalization=False,
		zca_whitening=False,
		zca_epsilon=1e-6,
		rotation_range=90,
		width_shift_range=0,
		height_shift_range=0,
		shear_range=0,
		zoom_range=0,
		channel_shift_range=0,
		fill_mode='nearest',
		cval=0.,
		horizontal_flip=True,
		vertical_flip=True,
		rescale=1.0/255.,
		preprocessing_function=None)
	return datagen_augmented, datagen

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.0001
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def train_and_validate(model, train_data_gen, valid_data_gen):
    train_samples_0 = len(os.listdir(tmp_train_data_path+"/label_3/"))
    train_samples_1 = len(os.listdir(tmp_train_data_path+"/label_45/"))  
    train_samples = train_samples_0 + train_samples_1
    
    val_samples=len(os.listdir(val_data_path+"/label_3/"))+len(os.listdir(val_data_path+"/label_45/"))
    
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

    lrate = LearningRateScheduler(step_decay)
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
            callbacks=[checkpointer, csv_logger, lrate],
            class_weight=class_weight)

##################################################################
#--------------------- EM Algo Helper functions -------------------#
##################################################################
def load_indices(data_path):
    img_wise_indices = []
    patch_wise_indices = []

    for label in labels_list:
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

    for label in labels_list:
        class_data_path = os.path.join(tmp_train_data_path, "label_"+str(label))
        patch_list = os.listdir(class_data_path)

        class_probab_map = np.zeros((1,))
        class_probab_map = np.delete(class_probab_map, [0], axis=0)

        # Iterating img_wise over the patches
        for img_name in img_wise_indices[labels_list.index(label)].keys():
            patches = np.zeros((1, img_rows, img_cols, 3))
            patches = np.delete(patches, [0], axis=0)
            
            # For all the patches of the image
            for patch_index in img_wise_indices[labels_list.index(label)][img_name]:
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

    for label in labels_list:
        class_probab_map = np.load(os.path.join(probab_path, "label_"+str(label)+".npy"))
        class_lvl_thresh = np.percentile(class_probab_map, class_lvl_pctl)

        class_data_path = os.path.join(train_data_path, "label_"+str(label))
        patch_list = os.listdir(class_data_path)

        for img_name in img_wise_indices[labels_list.index(label)].keys():
            img_probab_map = np.load(os.path.join(probab_path, "label_"+str(label), img_name+".npy"))
            img_lvl_thresh = np.percentile(img_probab_map, img_lvl_pctl)
            threshold = min(img_lvl_thresh, class_lvl_thresh)

            reconstructed_probab_map = np.ones([2000, 2000])*threshold
            
            # Iterating over all the patches of the image
            for index in range(img_probab_map.shape[0]):
                patch_index = img_wise_indices[labels_list.index(label)][img_name][index]
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
                patch_index = img_wise_indices[labels_list.index(label)][img_name][index]
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
        for label in labels_list:
            os.mkdir(os.path.join(dir, "label_"+str(label)))

if __name__ == '__main__':
    main()
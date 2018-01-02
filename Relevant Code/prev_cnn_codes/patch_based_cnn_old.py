import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import scipy.stats as st
import tensorflow as tf
import cv2
import shutil

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras import backend as K


# --------------------- Global Variables -------------------------
# model_path = './models/patch_based_cnn_model'
model_path = './models'
patient_wise_patches_path = './tmp/anno_cent'
img_wise_patches_path = './tmp/img_wise_patches'
data_path = './tmp/data'

n_iter = 2

# @todo : code is yet to be made more efficient
# @todo : accuracy
# @todo : 3 classes rather than 2 classes
# @todo : Saving and loading the model every iteration
# @todo : Description


"""
	patches[0][img_name]={	"patches" : All the patches as stacked numpy arrays 
							"coord"	  : Center coords of each patch}	
"""

# --------------------- Main Function ----------------------------
def main():
	""" Trains the model on EM algo and saves it to the given model path """
	# Default Session for Keras
	sess = tf.Session()
	K.set_session(sess)

	# Sorts all the patches into image_wise directories
	# print("Loading img_wise patches now : ")
	# load_patches_image_wise(patient_wise_patches_path, img_wise_patches_path)
	# print("Loading patches imgwise completed!!")

	# # Prepares training data for first iteration in the training direct batch wise
	print("init data being loaded now :")
	init_data_load(img_wise_patches_path, data_path, val_test_batch_size=1500)
	print("init data loaded!!!")

	#################### Initial M-step ######################## 
	# Training and predictions of probability maps
	preds, accuracy_metric, img, labels, train_step = patch_based_cnn_model(sess)
	train(sess, data_path, img, labels, train_step, n_epochs=1)												#FIXME
	print("First iteration of EM algo over")

	# Validation part
	validate(sess, accuracy_metric, data_path, img, labels)

	#################### 2nd Iteration Onwards ########################
	train_img_wise_patches_path = os.path.join(img_wise_patches_path, "train")
	for itr in range(n_iter-1):
		# E-step
		generate_predicted_maps(sess, preds, img_wise_patches_path, img=img)
		print("Probab maps predicted!!")

		E_step(img_wise_patches_path, data_path)
		print("{}th iteration's E-step performed!!".format(itr+1))
		
		# M-Step
		train(sess, data_path, img, labels, train_step)
		print("{}th iteration's M-step performed!!".format(itr+1))

		# Validation part
		validate(sess, accuracy_metric, data_path, img, labels)

	# saving the model
	saver = tf.train.Saver()
	saver.save(sess, model_path)

	# EM-Algo completed
	sess.close()
	print("Training Completed!!")



##################################################################
#---------------------Model and its functions -------------------#
##################################################################
def patch_based_cnn_model(sess, dropout_prob=0.5, l_rate=0.5, n_classes=2):

	# Placeholders
	img = tf.placeholder(tf.float32, shape=(None, 101, 101, 3))
	labels = tf.placeholder(tf.float32, shape=(None, 2))

	# Layers
	convnet = Conv2D(80, 6, strides=1, padding='valid', activation=None, kernel_initializer='he_normal')(img)
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

	convnet = tf.reshape(convnet, [-1, 9*9*200])
	convnet = Dense(320, activation='relu')(convnet)
	convnet = Dropout(dropout_prob)(convnet)

	convnet = Dense(320, activation='relu')(convnet)
	convnet = Dropout(dropout_prob)(convnet)

	preds = Dense(n_classes, activation='linear')(convnet)
	preds = Activation('softmax')(preds)

	sess.run(tf.global_variables_initializer())

	# loss funtion
	loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

	# Training operation
	train_step = tf.train.GradientDescentOptimizer(l_rate).minimize(loss)

	# Accurace metric
	accuracy_metric = tf.reduce_mean(accuracy(labels, preds))

	return [preds, accuracy_metric, img, labels, train_step]


def train(sess, data_path, img, labels, train_step, n_epochs=1):
	train_data_path = os.path.join(data_path, "train")
	batch_list = os.listdir(train_data_path)
	for epoch in range(n_epochs):
		for batch_number in range(len(batch_list)):
			patches_file = os.path.join(train_data_path, "batch_{}".format(batch_number+1), "patches.npy")
			labels_file = os.path.join(train_data_path, "batch_{}".format(batch_number+1), "label.npy")

			source_patches = np.load(patches_file)
			source_labels = np.load(labels_file)

			# print("Patches' shape :", source_patches.shape)
			# print("Labels' shape :", source_labels.shape)

			sess.run(train_step, feed_dict={img: source_patches,											
											labels: source_labels,
											K.learning_phase(): 1})

			# print("Batch : " +  "{}".format(batch_number+1) + " running........")
		print("Epoch : " + str(epoch+1) + " completed!!")

def validate(sess, accuracy_metric, data_path, img, labels):							#FIXME
	val_data_path = os.path.join(data_path, "val")
  	val_batch_dirlist = os.listdir(val_data_path)
  	acc = 0
	for batch_dirname in val_batch_dirlist:
		patches_file = os.path.join(val_data_path, batch_dirname, "patches.npy")
		labels_file = os.path.join(val_data_path, batch_dirname, "label.npy")

		patches = np.load(patches_file)
		patch_labels = np.load(labels_file)

		acc += sess.run(accuracy_metric, feed_dict={  img: patches,
		               		               		      labels: patch_labels,
		                    	                	  K.learning_phase(): 0})
	print("Accuracy list:")
	print(acc)
	print("The accuracy of the model is :", acc/len(val_batch_dirlist))					# Assuming every batch of same size


##################################################################
#--------------------- EM Algo Helper functions -------------------#
##################################################################
def load_patches_image_wise(cent_patches_dir, img_wise_patches_path, n_classes=2):
	""" - Loads all the images and splits them into patches for prediction
		and updation of training data for every iteration in the applied
		EM algo for training he model
		- Returns:
			All the patches arranged in a dict img_wise (will be required when reconstructin the probability map)

	"""

	one_hot_vec = np.zeros(shape=(1, n_classes))
	train_img_wise_patches_path = os.path.join(img_wise_patches_path, "train")
	test_img_wise_patches_path = os.path.join(img_wise_patches_path, "test") 
	val_img_wise_patches_path = os.path.join(img_wise_patches_path, "val")

	# Ensures the dir presence and is empty
	remove_and_create_dir(img_wise_patches_path)
	remove_and_create_dir(train_img_wise_patches_path)
	remove_and_create_dir(test_img_wise_patches_path)
	remove_and_create_dir(val_img_wise_patches_path)

	for i in range(n_classes):																	# iterated over label 1 and 0
		# Load train, test and val patient ids list
		train_list = []
		test_list = []
		val_list = []

		with open("train_{}.txt".format(i)) as myfile:
			train_list = myfile.readlines()

		with open("val_{}.txt".format(i)) as myfile:
			val_list = myfile.readlines()

		with open("test_{}.txt".format(i)) as myfile:
			test_list = myfile.readlines()

		one_hot_vec[0,i] = 1
		labeled_patches_dir = os.path.join(cent_patches_dir, 'label_'+str(i))
		patient_wise_patches_dirlist = os.listdir(labeled_patches_dir)
		print("Starting for label ", i)
		for patient_dirname in patient_wise_patches_dirlist:
			patient_id = patient_dirname.split("_")[0]
			patient_dir_path = os.path.join(labeled_patches_dir, patient_dirname)
			patch_list = os.listdir(patient_dir_path)
			img_wise_patches = {}
			
			if (patient_id + "\n") in train_list:
				data_split = "train"
			elif (patient_id + "\n") in test_list:
				data_split = "test"
			elif (patient_id + "\n") in val_list:
				data_split = "val"
			else:
				continue

			for patchname in patch_list:
				patch_path = os.path.join(patient_dir_path, patchname)
				
				patch_name_split = patchname.split("_")											# 1000104570_999_913_PrognosisTMABlock3_F_4_5_H&E0.png
				img_name = "_".join([patch_name_split[0],]+patch_name_split[3:]).split(".")[0]	# img_name = "1000104570_PrognosisTMABlock3_F_4_5_H&E0_%d".format(i)
				img_name += "_{}".format(i)
				patch_coord = np.array([[int(patch_name_split[1]), int(patch_name_split[2])]])	# patch_coord as numpy arrays of shape = (1,2)
				
				patch = load_patch(patch_path)
				patch = np.reshape(patch, (1, 101, 101, 3))

				if img_name in img_wise_patches.keys():
					img_wise_patches[img_name]["patches"] = np.concatenate((img_wise_patches[img_name]["patches"], patch))
					img_wise_patches[img_name]["coord"]  = np.concatenate((img_wise_patches[img_name]["coord"], patch_coord))
					img_wise_patches[img_name]["label"] = np.concatenate((img_wise_patches[img_name]["label"], one_hot_vec))
				else:
					img_wise_patches[img_name] = {}
					img_wise_patches[img_name]["label"] = one_hot_vec 
					img_wise_patches[img_name]["patches"] = patch
					img_wise_patches[img_name]["coord"] = patch_coord
					img_wise_patches[img_name]["data_split"] = data_split
			
			for img_name in img_wise_patches.keys():
				img_dir = os.path.join(img_wise_patches_path, img_wise_patches[img_name]["data_split"], img_name)
				if not os.path.exists(img_dir):
					os.mkdir(img_dir)
					patches_file = os.path.join(img_dir, "patches")
					label_file = os.path.join(img_dir, "label")
					coord_file = os.path.join(img_dir, "coord")

					np.save(patches_file, img_wise_patches[img_name]["patches"])
					np.save(label_file, img_wise_patches[img_name]["label"])
					np.save(coord_file, img_wise_patches[img_name]["coord"])

			print("Completed for patient :", patient_id)

	print("Patches Extraction Completed!!")


def init_data_load(img_wise_patches_path, dest_data_path, batch_size=128, val_test_batch_size=1500):		# dest = destination
	data_split = ['train', 'test', 'val']
	delete_mask = list(range(batch_size))

	# Esure the dir presenec for data storage
	remove_and_create_dir(dest_data_path)

	for split in data_split:																# iterating over every split
		batch_number = 1

		split_img_wise_patches_path = os.path.join(img_wise_patches_path, split)
		split_data_path = os.path.join(dest_data_path, split)

		# Checks if the dir is present and ensures it is empty for the new data being pumped in
		remove_and_create_dir(split_data_path)

		img_list = os.listdir(split_img_wise_patches_path)

		patches = np.zeros((1,101,101,3))
		patches = np.delete(patches, [0], axis=0)

		labels = np.zeros((1,2))
		labels = np.delete(labels, [0], axis=0)
		
		if split is 'train':
			split_batch_size = batch_size
		else:
			split_batch_size = val_test_batch_size

		for img_name in img_list:
			source_label_file = os.path.join(split_img_wise_patches_path, img_name, "label.npy")
			source_coord_file = os.path.join(split_img_wise_patches_path, img_name, "coord.npy")
			source_patches_file = os.path.join(split_img_wise_patches_path, img_name, "patches.npy")
			
			patches = np.concatenate((patches, np.load(source_patches_file)))
			labels = np.concatenate((labels, np.load(source_label_file)))

			while(1):
				batch_dir = os.path.join(split_data_path, "batch_{}".format(batch_number))
				dest_patches_file = os.path.join(batch_dir, "patches.npy")
				dest_label_file = os.path.join(batch_dir, "label.npy")
				# print(patches.shape)

				if patches.shape[0] >= split_batch_size:
					if not os.path.exists(batch_dir):
						os.mkdir(batch_dir)

					np.save(dest_patches_file, patches[0:split_batch_size])
					np.save(dest_label_file, labels[0:split_batch_size])

					patches = np.delete(patches, delete_mask, axis=0)
					labels = np.delete(labels, delete_mask, axis=0)
				else:
					if img_list[-1] is img_name:
						if not os.path.exists(batch_dir):
							os.mkdir(batch_dir)

						np.save(dest_patches_file, patches)
						np.save(dest_label_file, labels)						
					break

				print("{} batch {} completed".format(split, batch_number))
				batch_number += 1


def generate_predicted_maps(sess, preds, img_wise_patches_path, img, max_patches_per_prediction=1500):
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
def remove_and_create_dir(dir_path):
	if os.path.exists(dir_path):
		shutil.rmtree(dir_path)

	os.mkdir(dir_path)

def load_patch(img_path):
	return cv2.imread(img_path)

def GaussianKernel(ksize=101, nsig=30):
	gauss1D = cv2.getGaussianKernel(ksize, nsig)
	gauss2D = gauss1D*np.transpose(gauss1D)
	gauss2D = gauss2D/gauss2D[int(ksize/2), int(ksize/2)]
	return gauss2D

if __name__ == '__main__':
	main()


import numpy as np
import os
import scipy.stats as st
import tensorflow as tf

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras import backend as K

# --------------------- Global Variables -------------------------
# model_path = './models/patch_based_cnn_model'
patient_wise_patches_path = '../deepak/DB_HnE_101_anno_cent'
img_wise_patches_path = './img_wise_patches'
data_path = './data'

n_iter = 1															# number of iterations for EM algo to run

# @todo : Training, test and val set prep
# @todo : Evaluation in every iteration
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

	saver = tf.train.Saver()

	# Sorts all the patches into image_wise directories
	# patches, init_train_data = load_patches_image_wise(patches_path)
	if not os.path.exists(img_wise_patches_path):
		load_patches_image_wise(patches_path)

	# Prepares training data for first iteration in the training direct batch wise
	if not os.patch.exists(train_data_path):
		init_data_load(patches_path, train_path, test_path, val_path)

	#################### Initial M-step ######################## 
	# Training and predictions of probability maps
	preds, pred_class, loss, train_step = patch_based_cnn_model()
	for epoch in xrange(1, 10):																		#TODO
		epoch_loss = 0
		for i in range(int(train_x.shape[0]/batch_size)+1):
			if i == int(train_x.shape[0]/batch_size):
				sess.run(train_step, feed_dict={img: train_x[i*batch_size:,:,:,:],											
												labels: train_y[i*batch_size:,:],
												K.learning_phase(): 1})
			else:
				sess.run(train_step, feed_dict={img: train_x[i*batch_size:(i+1)*batch_size,:,:,:],
												labels: train_y[i*batch_size:(i+1)*batch_size,:],
												K.learning_phase(): 1})
		# Validation part 																			#TODO

		print("Epoch :", epoch, "loss is :", epoch_loss)

	#################### 2nd Iteration Onwards ########################
	for itr in range(n_iter-1):
		# E-step
		predicted_maps = []																			#TODO
		for i in range(2):
			labeled_predicted_maps = {}
			for image in patches[i].keys():
				labeled_predicted_maps[image] = sess.run(preds, feed_dict={	img: patches[i][image]["patches"],
																   			K.learning_phase(): 0})
		predicted_maps += [labeled_predicted_maps,]

		train_data = E_step(predicted_maps, patches, img_lvl_pctl, class_lvl_pctl)					#TODO

		# M-Step
		for epoch in xrange(1, n_epochs):
			epoch_loss = 0																			#TODO
			for i in range(int(train_x.shape[0]/batch_size)+1):
				if i == int(train_x.shape[0]/batch_size):
					sess.run(train_step, feed_dict={img: train_x[i*batch_size:,:,:,:],											
													labels: train_y[i*batch_size:,:],
													K.learning_phase(): 1})
				else:
					sess.run(train_step, feed_dict={img: train_x[i*batch_size:(i+1)*batch_size,:,:,:],
													labels: train_y[i*batch_size:(i+1)*batch_size,:],
													K.learning_phase(): 1})

			# print("Epoch :", epoch, "loss is :", epoch_loss)

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
	conv1 = tf.nn.local_response_normalisation(conv1)
	conv1 = Activation('relu')(conv1)
	conv1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

	conv2 = Conv2D(120, 5, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(conv1)
	conv2 = tf.nn.local_response_normalisation(conv2)
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
def load_patches_image_wise(cent_patches_dir, img_wise_patches_path, n_classes=2):
	""" - Loads all the images and splits them into patches for prediction
		and updation of training data for every iteration in the applied
		EM algo for training he model
		- Returns:
			All the patches arranged in a dict img_wise (will be required when reconstructin the probability map)

	"""

	# img_wise_patches = []
	# init_train_data = [1, 2]																	# Initialised for a check as a list of int
	one_hot_vec = np.zeros(shape=(1, n_classes))


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
		# labeled_img_wise_patches = {}
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

				# if type(init_train_data[0]) is np.ndarray:
				# 	init_train_data[0] = np.concatenate((init_train_data[0], patch))
				# 	init_train_data[1] = np.concatenate((init_train_data[1], one_hot_vec))
				# else:
				# 	init_train_data[0] = patch
				# 	init_train_data[1] = one_hot_vec

		# img_wise_patches += [labeled_img_wise_patches,]

	# print(len(img_wise_patches[0]))
	# print(init_train_data[0].shape)
	# print(init_train_data[1].shape)

	# return [img_wise_patches, init_train_data]
	print("Patches Extraction Completed!!")

def init_data_load(img_wise_patches_path, dest_data_path, batch_size=128):		# dest = destination
	data_split = ['train', 'test', 'val']
	batch_number = 1
	delete_mask = list(range(batch_size))
	for split in data_split:													# iterating over every split
		split_img_wise_patches_path = os.path.join(img_wise_patches_path, split)
		split_data_path = os.path.join(dest_data_path, split)
		img_list = os.listdir(split_img_wise_patches_path)
		
		patches = np.zeros((1,101,101,3))
		patches = np.delete(patches, [0], axis=0)

		labels = np.zeros((1,2))
		labels = np.delete(labels, [0], axis=0)

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
				print(patches.shape)

				if patches.shape[0] >= batch_size:
					if not os.path.exists(batch_dir):
						os.mkdir(batch_dir)

					np.save(dest_patches_file, patches[0:batch_size])
					np.save(dest_label_file, labels[0:batch_size])

					patches = np.delete(patches, delete_mask, axis=0)
					labels = np.delete(labels, delete_mask, axis=0)
				else:
					if img_list[-1] is img_name:
						os.mkdir(batch_dir)

						np.save(dest_patches_file, patches)
						np.save(dest_label_file, labels)						
					break

				batch_number += 1

def E_step(predicted_maps, img_wise_patches, img_lvl_pctl=30, class_lvl_pctl=30, n_classes=2):
	""" - Applies gaussian smoothing to the predicted probability maps
		- Generates bit mask for acceptable()discriminative patches for the next M_step
		- Returns:
				Refined training data for the next step
	"""
	new_train_data = [1, 2]															# Initialised for a check as a list of int
	one_hot_vec = np.zeros(shape=(1, n_classes))

	for i in range(n_classes):																# Iterating over all the labels
		one_hot_vec[0,i] = 1
		class_prob_map = []
		
		for img_name in predicted_maps[i].keys():
			if type(class_prob_map) is np.ndarray:
				class_prob_map = np.concatenate((class_prob_map, predicted_maps[i][img_name]))
			else:
				class_prob_map = predicted_maps[i][img_name]

		for img_name in predicted_maps[i].keys():											# iterating over images
			reconstructed_probab_map = np.zeros([101,101])

			for j in range(len(patches[i][img_name]["coord"])):								# iterating over patches
				center_coord = patches[i][img_name]["coord"][j]
				probability = predicted_maps[i][img_name][j,i]
				orig_patch_probab = reconstructed_probab_map[center_coord[0]-50: center_coord[0]+50, center_coord[1]-50:center_coord[1]+50]
				new_patch_prob = probability*np.GaussianKernel()
				reconstructed_probab_map[center_coord[0]-50: center_coord[0]+50, center_coord[1]-50:center_coord[1]+50] = 
						np.maximum(orig_patch, new_patch_prob)

			img_lvl_thresh = np.percentile(predicted_maps[i][img_name][:,i], axis=0)
			class_lvl_thresh = np.percentile(class_prob_map, axis=0)

			threshold = min(img_lvl_thresh, class_lvl_thresh)
			discriminative_mask = (reconstructed_probab_map>threshold) | (reconstructed_probab_map==threshold)
			
			for j in range(len(patches[i][img_name]["coord"])):
				coord = patches[i][img_name]["coord"][i]
				if discriminative_mask[coord[0], coord[1]]:
					patch = np.reshape(patches[i][img_name]["patches"][j], [1, 101, 101, 3])
					if type(new_train_data[0]) is np.ndarray:
						new_train_data[0] = np.concatenate((new_train_data[0], patch))
						new_train_data[1] = np.concatenate((new_train_data[1], one_hot_vec))
					else:
						new_train_data[0] = patch
						new_train_data[1] = one_hot_vec

	return new_train_data


#################################################################
# -------------------- Other Helper functions ------------------#
#################################################################
def load_patch(img_path):
	return cv2.imread(img_path)

def GaussianKernel(ksize=100, nsig=30):
	gauss1D = cv2.getGaussianKernel(ksize, nsig)
	gauss2D = gauss1D*np.transpose(gauss1D)
	gauss2D = gauss2D/gauss2D[int(ksize/2), int(ksize/2)]
	return gauss2D


if __name__ == '__main__':
	main()


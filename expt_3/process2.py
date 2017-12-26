import cv2
import numpy as np
import os
from os.path import join

# GLobal Variables
dir_path = './valid'
n_classes = 2

def main():
	img_wise_indices, patch_wise_indices = load_indices(dir_path)

	for label in range(n_classes):
		label_dir_path = join(dir_path, "label_"+str(label))
		patches_list = os.listdir(label_dir_path)
	
		patches = np.zeros((1, 51, 51, 3))
		patches = np.delete(patches, [0], axis=0)
	
		for img_name in img_wise_indices[label].keys():
			for index in img_wise_indices[label][img_name]:
				patch_path = join(label_dir_path, patches_list[index])
				patch = load_patch(patch_path)
				patches = np.concatenate((patches, np.expand_dims(patch, axis=0)))
				os.remove(patch_path)
			np.save(join(label_dir_path, img_name+".npy"), patches)

# ---------------- Helper Functions
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
 			#img_name = patch_split[0]
 			#img_size = [int(patch_split[4]), int(patch_split[5])]
 			if img_name not in label_img_wise_indices.keys():
 				label_img_wise_indices[img_name] = [patch_index,]
 			else:
 				label_img_wise_indices[img_name] += [patch_index,]

 			label_patch_wise_indices[patch_name]={}
 			label_patch_wise_indices[patch_name]["index"]=patch_index
 			label_patch_wise_indices[patch_name]["img_name"]=img_name
 			label_patch_wise_indices[patch_name]["coord"]=[int(patch_split[1]), int(patch_split[2])]
	 		#label_patch_wise_indices[patch_name]["img_shape"]=img_size

		img_wise_indices += [label_img_wise_indices,]
		patch_wise_indices += [label_patch_wise_indices,]

	return img_wise_indices, patch_wise_indices

def load_patch(img_path):
	return cv2.imread(img_path)

if __name__=="__main__":
	main()

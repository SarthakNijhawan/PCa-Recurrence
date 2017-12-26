import shutil
import os
import cv2
import numpy as np

# GLobal Variables
src = "./PetImages"
dest = "./data_dump/"

n_classes = 2
img_channels = 3
patch_size = 75
strides = 35

types = ["train_0", "train_1", "valid_0", "valid_1"]

def main():
	make_dir(dest, False)
	
	for data_type in types:
		print(data_type)
		file = open(data_type+'.txt')
		lines = file.readlines()
		type_dest = os.path.join(dest, data_type.split("_")[0])
		label = int(data_type.split("_")[1])
		if not os.path.exists(type_dest):
			make_dir(type_dest)
			
		for line in lines:
			img_name = line.split("\n")[0]
			image_path = os.path.join(src, "label_"+str(label), img_name+".jpg")
			
			patches, coord_list, img_size, proceed = make_patches(image_path)

			if proceed == True:
				for i in range(patches.shape[0]):
					x_coord, y_coord = coord_list[i]
					patch_dest_path = os.path.join(type_dest, "label_"+str(label), img_name+"_"+str(x_coord)+"_"+str(y_coord)+"_"+str(label)+"_"+str(img_size[0])+"_"+str(img_size[1])+".png")
					cv2.imwrite(patch_dest_path, patches[i])
			else:
				continue
		file.close()

def make_patches(image_path):
	image = load_image(image_path)
	#print(image.shape)
	try:
		img_size = image.shape
		if img_size[0] < 450 or img_size[1] < 450 :
			print(image_path, "is skipped....")
			return -1, -1, -1, False
	except AttributeError:
		print(image_path, "is corrupted..")
		return -1, -1, -1, False

	patches = np.zeros((1, patch_size, patch_size, img_channels))
	patches = np.delete(patches, [0], axis=0)
	coord_list = []
	
	for x_coord in range(patch_size/2, img_size[0]-patch_size/2, strides):
		for y_coord in range(patch_size/2, img_size[1]-patch_size/2, strides):
			patch = image[x_coord-patch_size/2:x_coord+patch_size/2+1, y_coord-patch_size/2:y_coord+patch_size/2+1]
			# print(patches.shape)
			# print(patch.shape)
			patches = np.concatenate((patches, np.expand_dims(patch, axis=0)))
			coord_list += [[x_coord, y_coord],]

	return patches, coord_list, img_size, True

def load_image(img_path):
	return cv2.imread(img_path)

def make_dir(dir, needlabel=True):
	if os.path.exists(dir):
		shutil.rmtree(dir)
	os.mkdir(dir)
	if needlabel is True:
		for label in range(n_classes):
			os.mkdir(os.path.join(dir, "label_"+str(label)))

if __name__ == '__main__':
	main()

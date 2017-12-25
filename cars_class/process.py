import shutil
import os
import cv2
import numpy as np

src = "./train_images"
dest = "./data_dump/"

patch_size = (51, 51, 3)

types = ["train", "valid"]

def main():
	make_dir(dest, False)

	for data_type in types:
		file = open(data_type+'_anno.txt')
		lines = file.readlines()
		type_dest = os.path.join(dest, data_type)
		make_dir(type_dest)

		for line in lines:
			line_split = line.split("\n")[0].split(" ")
			image_path = os.path.join(src, line_split[1])
			img_name = line_split[1].split(".")[0]
			label = int(line_split[0])-1
			patches, coord_list, img_size = make_patches(image_path)

			for i in range(patches.shape[0]):
				x_coord, y_coord = coord_list[i]
				patch_dest_path = os.path.join(type_dest, "label_"+str(label), img_name+"_"+str(x_coord)+"_"+str(y_coord)+"_"+str(label)+"_"+str(img_size[0])+"_"+str(img_size[1])+".png")
				cv2.imwrite(patch_dest_path, patches[i])
			print("Processing done for :", img_name)

	file.close()

def make_patches(image_path, patch_size=51, strides=23):
	image = load_image(image_path)
	print(image.shape)
	img_size = image.shape
	patches = np.zeros((1, 51, 51, 3))
	patches = np.delete(patches, [0], axis=0)
	coord_list = []
	
	for x_coord in range(patch_size/2, img_size[0]-patch_size/2, strides):
		for y_coord in range(patch_size/2, img_size[1]-patch_size/2, strides):
			patch = image[x_coord-patch_size/2:x_coord+patch_size/2+1, y_coord-patch_size/2:y_coord+patch_size/2+1]
			# print(patches.shape)
			# print(patch.shape)
			patches = np.concatenate((patches, np.expand_dims(patch, axis=0)))
			coord_list += [[x_coord, y_coord],]

	return patches, coord_list, img_size

def load_image(img_path):
	return cv2.imread(img_path)

def make_dir(dir, needlabel=True, n_classes=2):
	if os.path.exists(dir):
		shutil.rmtree(dir)
	os.mkdir(dir)
	if needlabel is True:
		for label in range(n_classes):
			os.mkdir(os.path.join(dir, "label_"+str(label)))

if __name__ == '__main__':
	main()
import shutil
import os
import numpy as np
import cv2
from xml.dom import minidom
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from sklearn.feature_extraction.image import extract_patches_2d

seed = np.random.RandomState(26)
width = 24
height = 24
angle = 180
sig = 4
alpha = 20
patch_size = np.array([2000, 2000])
stride = 31

def elastic_transform(medical, bit_mask, bit_bound, alpha, sigma, random_state=None, number=1):
	"""Elastic deformation of images as described in [Simard2003]_.
	.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
	   Convolutional Neural Networks applied to Visual Document Analysis", in
	   Proc. of the International Conference on Document Analysis and
	   Recognition, 2003.
	"""
	assert len(medical.shape)==2

	if random_state is None:
		random_state = np.random.RandomState(None)

	shape = medical.shape

	medical_list = list()
	bit_mask_list = list()
	bit_bound_list = list()
	for i in range(0, number):

		dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
		dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

		x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
		indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

		medical_list.append(map_coordinates(medical, indices, order=1).reshape(shape))
		bit_mask_list.append(map_coordinates(bit_mask, indices, order=1).reshape(shape))
		bit_bound_list.append(map_coordinates(bit_bound, indices, order=1).reshape(shape))

	return (medical_list, bit_mask_list, bit_bound_list)

def rotation_transform(medical, bit_mask, bit_bound, angle, random_state=None, number=1):
	"""
	Uniformly picks a angle in [-angle, angle] and rotates the image
	in clockwise or direction. Same rotation is applied to the 
	bit_mask as well. 
	Returns: A list in which the first entry is a list of medical images,
	second is the corresponding bit_mask  
	"""
	assert len(medical.shape)==2

	if random_state is None:
		random_state = np.random.RandomState(None)

	rows, cols = medical.shape

	medical_list = list()
	bit_mask_list = list()
	bit_bound_list = list()
	for i in range(0, number):

		angle_rotate = random_state.uniform(-angle, angle) # Sampling the angle
		Rotation_M = cv2.getRotationMatrix2D((cols/2, rows/2), angle_rotate, 1) # Creating the rotation matrix
		medical_list.append(cv2.warpAffine(medical, Rotation_M, (cols, rows))) # Applying the transformation
		bit_mask_list.append(cv2.warpAffine(bit_mask, Rotation_M, (cols, rows))) # Applying the same transformation
		bit_bound_list.append(cv2.warpAffine(bit_bound, Rotation_M, (cols, rows))) # Applying the same transformation

	return (medical_list, bit_mask_list, bit_bound_list)

def translation_transform(medical, bit_mask, bit_bound, width, height, random_state=None, number=1):
	"""
	Uniformly picks a height, width in [-height, height], [-width, width]
	resp. and translates the image by that much amount. Same translation is 
	applied to the bit_mask as well.
	Returns: A list in which the first entry is a list of medical images,
	second is the corresponding bit_mask 
	"""	
	assert len(medical.shape)==2

	if random_state is None:
		random_state = np.random.RandomState(None)

	rows, cols = medical.shape

	medical_list = list()
	bit_mask_list = list()
	bit_bound_list = list()
	for i in range(0, number):

		tr_x = random_state.uniform(-height, height)
		tr_y = random_state.uniform(-width, width)
		Translation_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
		medical_list.append(cv2.warpAffine(medical, Translation_M, (cols, rows)))
		bit_mask_list.append(cv2.warpAffine(bit_mask, Translation_M, (cols, rows)))
		bit_bound_list.append(cv2.warpAffine(bit_bound, Translation_M, (cols, rows)))

	return (medical_list, bit_mask_list, bit_bound_list)

def flips(medical, bit_mask, bit_bound, number=1):
	"""
	Gives a vertically flipped image or horizontally.
	No point in giving number > 2.
	Returns: A list in which the first entry is a list of medical images,
	second is the corresponding bit_mask 
	"""
	assert len(medical.shape)==2

	medical_list = list()
	bit_mask_list = list()
	bit_bound_list = list()
	for i in range(0, number):
		medical_list.append(cv2.flip(medical, i%2))
		bit_mask_list.append(cv2.flip(bit_mask, i%2))
		bit_bound_list.append(cv2.flip(bit_bound, i%2))

	return (medical_list, bit_mask_list, bit_bound_list)	

def data_augment(img_list, bit_list, bound_list):
	"""
	Function assumes that both are grey scale images.
	Applies the same random rotation, translation, flips and 
	elastic deforation to both medical and corresponding bit_mask.
	The arguments are LISTS.
	"""
	medical_list = list()
	bit_mask_list = list()
	bit_bound_list = list()

	for m_img, b_img, bound_img in zip(img_list, bit_list, bound_list):
		m_flist, b_flist, bound_flist = flips(m_img, b_img, bound_img, 2)
		m_rlist, b_rlist, bound_rlist = rotation_transform(m_img, b_img, bound_img, angle, seed, 2)
		m_tlist, b_tlist, bound_tlist = translation_transform(m_img, b_img, bound_img, width, height, seed, 2)
		m_elist, b_elist, bound_elist = elastic_transform(m_img, b_img, bound_img, sig, alpha, seed, 2)
		medical_list += m_flist + m_rlist + m_tlist + m_elist 
		bit_mask_list += b_flist + b_rlist + b_tlist + b_elist
		bit_bound_list += bound_flist + bound_rlist + bound_tlist + bound_elist

	return (medical_list, bit_mask_list, bit_bound_list)

def generate_bit_mask(shape, xml_file):
	"""
	Given the image shape and path to annotations(xml file), 
	generate a bit mask with the region inside a contour being white
	shape: The image shape on which bit mask will be made
	xml_file: path relative to the current working directory 
	where the xml file is present
	Returns: A image of given shape with region inside contour being white..
	"""
	# DOM object created by the minidom parser
	xDoc = minidom.parse(xml_file)

	# List of all Region tags
	regions = xDoc.getElementsByTagName('Region')

	# List which will store the vertices for each region
	xy = []
	for region in regions:
		# Loading all the vertices in the region
		vertices = region.getElementsByTagName('Vertex')

		# The vertices of a region will be stored in a array
		vw = np.zeros((len(vertices), 2))

		for index, vertex in enumerate(vertices):
			# Storing the values of x and y coordinate after conversion
			vw[index][0] = float(vertex.getAttribute('X'))
			vw[index][1] = float(vertex.getAttribute('Y'))
		x_series = vw[:,0]
		y_series = vw[:,1]
		# print(x_series
		avg_x=np.mean(x_series)
		avg_y=np.mean(y_series)
		# print(avg_x, avg_y
		new_coord_x=x_series-avg_x
		new_coord_y=y_series-avg_y
		new_coord_x=.5*new_coord_x
		new_coord_y=.5*new_coord_y
		new_coord_x=new_coord_x+avg_x
		new_coord_y=new_coord_y+avg_y

		# print(x_series-new_coord_x
		# print(y_series - new_coord_y
		vw[:,0]=new_coord_x
		vw[:,1]=new_coord_y		


		# Append the vertices of a region
		xy.append(np.int32(vw))

	# Creating a completely black image
	mask = np.zeros(shape, np.uint8)
	# mask for boundaries
	mask_boundary = np.zeros(shape, np.uint8)

	# For each contour, fills the area inside it
	# Warning: If a list of contours is passed, overlapping regions get buggy output
	# Comment out the below line to check, and if the bug is fixed use this
	# cv2.drawContours(mask, xy, -1, (255,255,255), cv2.FILLED)
	for contour in xy:
		cv2.drawContours(mask, [contour], -1, (255,255,255), cv2.FILLED)
		cv2.drawContours(mask_boundary, [contour], -1, (255,255,255), 3)
	print(np.unique(mask))
	return mask, mask_boundary

def data_mirror(image_list, width, height):
	"""
	Takes a list of images and extends the images on all sides.
	If the shape of image is rows, cols then afterwards it will be
	rows+2*height, cols+2*width.
	The extension is done if form of mirroring the border of the image. 
	"""
	mirror_list = [cv2.copyMakeBorder(img,height,height,width,width,cv2.BORDER_REFLECT) for img in image_list]
	
	return mirror_list		

def create_patch(path_anot, path_img, save_path_anot, save_path_img, save_path_weight, patch_size, img_name, stride, is_img_aug=False):
	"""
	Goes through the files in path_anot as well as path_img, loads the image
	from path_img and generates the bit_mask from path_anot. Goes around the 
	image in both to extract patches of size specified above.
	Saves them into the location given by save_path.
	The path is a LIST of corresponding path names.
	"""
	j = 0
	for p_a, p_i, img_n in zip(path_anot, path_img, img_name):
		print(j)
		print(p_i)
		j += 1
		img = cv2.imread(p_i)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#####	
		#img = cv2.resize(img, (2000,200), interpolation = cv2.INTER_AREA)


		i_h, i_w = img.shape
		p_h, p_w = patch_size

		bit_m, bit_b = generate_bit_mask((i_h,i_w), p_a)
		#####
		#bit_m = cv2.resize(bit_m, (512,512), interpolation = cv2.INTER_AREA)
		#bit_b = cv2.resize(bit_b, (512,512), interpolation = cv2.INTER_AREA)

		assert img.shape == bit_m.shape
		i = 0 
		for x in range(0, i_w - p_w + 1, stride):
			for y in range(0, i_h - p_h + 1, stride):
				patch_img = img[x:x+p_w, y:y+p_h]
				patch_bit = bit_m[x:x+p_w, y:y+p_h]
				patch_bound = bit_b[x:x+p_w, y:y+p_h]

				if is_img_aug:
					patch_img_aug, patch_bit_aug, patch_bound_aug = data_augment([patch_img], [patch_bit], [patch_bound])
				else:
					patch_img_aug = [patch_img]
					patch_bit_aug = [patch_bit]
					patch_bound_aug = [patch_bound]
				# patch_img_mirror = data_mirror(patch_img_aug, width, height)

				for patch_imgf, patch_bitf, patch_boundf in zip(patch_img_aug, patch_bit_aug, patch_bound_aug):
					ret1, patch_bitf = cv2.threshold(patch_bitf, 127, 255, cv2.THRESH_BINARY)
					ret2, patch_boundf = cv2.threshold(patch_boundf, 127, 255, cv2.THRESH_BINARY)

					cv2.imwrite(save_path_img + img_n + "_" + str(i) + ".tif", patch_imgf)
					cv2.imwrite(save_path_anot + img_n + "_" + str(i) + ".tif", patch_bitf)
					#distance_trans = cv2.distanceTransform(patch_bitf, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
					####addition of thresholding
					# ret, sure_fg = cv2.threshold(distance_trans,0.3*distance_trans.max(),50,1)
					##########
					# max_dist = np.max(distance_trans)
					#cv2.imwrite(save_path_weight + "1/" + img_n + "_" + str(i) + ".tif", np.uint8(distance_trans))
					#ret_new, sure_fg = cv2.threshold(patch_bitf,127,12,cv2.THRESH_BINARY)
					#ret_new, sure_bg = cv2.threshold(patch_bitf,127,3,cv2.THRESH_BINARY_INV)
					#sure_fg = sure_fg+sure_bg

					#cv2.imwrite(save_path_weight + "1_20/" + img_n + "_" + str(i) + ".tif", np.uint8(sure_fg))
					# bound_dist = distance_trans + np.where(patch_boundf == 255, max_dist, 0)
					# cv2.imwrite(save_path_weight + "2_20/" + img_n + "_" + str(i) + ".tif", np.uint8(bound_dist))

					i += 1

def make_directory_structure(dir_img, dir_xml, save_path_anot, save_path_img, save_path_weight, is_img_aug=False):
	print(dir_img, dir_xml, save_path_anot, save_path_img, save_path_weight)
	if os.path.exists(save_path_anot):
		shutil.rmtree(save_path_anot)
	os.makedirs(save_path_anot)

	if os.path.exists(save_path_img):
		shutil.rmtree(save_path_img)
	os.makedirs(save_path_img)

	if os.path.exists(save_path_weight + "1_20/"):
		shutil.rmtree(save_path_weight + "1_20/")
	os.makedirs(save_path_weight + "1_20/")

	# if os.path.exists(save_path_weight + "2_20/"):
	# 	shutil.rmtree(save_path_weight + "2_20/")
	# os.makedirs(save_path_weight + "2_20/")

	path_anot = []
	path_img = []
	img_name_list = []

	for img_name, xml_name in zip(os.listdir(dir_img), os.listdir(dir_xml)):
		img_name = img_name.split('.')[0]
		path_anot.append(dir_xml + "/" + img_name + ".xml")
		path_img.append(dir_img  + "/" + img_name + ".tif")
		img_name_list.append(img_name)

	create_patch(path_anot[1:], path_img[1:], save_path_anot, save_path_img, save_path_weight, patch_size, img_name_list[1:], stride, is_img_aug=is_img_aug)

if __name__ == '__main__':
	cur_dir = os.getcwd()
	save_path_anot = os.path.join(cur_dir, "%s/%s_y/")
	save_path_img = os.path.join(cur_dir, "%s/%s_x/")
	save_path_weight = os.path.join(cur_dir, "%s/weights")
	dir_img = os.path.join(cur_dir, "%s/Tissue_images") 
	dir_xml = os.path.join(cur_dir, "%s/Annotations")
	# Reading the whole slide image using opencv

	make_directory_structure(dir_img % "Training", dir_xml % "Training", save_path_anot % ("Training", "Train_20"),\
	 save_path_img % ("Training", "Train_20"), save_path_weight % "Training")

	# make_directory_structure(dir_img % "Testing", dir_xml % "Testing", save_path_anot % ("Testing", "Test_20"),\
	#  save_path_img % ("Testing", "Test_20"), save_path_weight % "Testing", False)

	# make_directory_structure(dir_img % "Validation", dir_xml % "Validation", save_path_anot % ("Validation", "Valid_20"),\
	#  save_path_img % ("Validation", "Valid_20"), save_path_weight % "Validation", False)


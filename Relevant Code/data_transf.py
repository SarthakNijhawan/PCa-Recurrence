# -*- coding: utf-8 -*-
import sys
import os

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
	str_format = "{0:." + str(decimals) + "f}"
	percents = str_format.format(100 * (iteration / float(total)))
	filled_length = int(round(bar_length * iteration / float(total)))
	bar = '█' * filled_length + '-' * (bar_length - filled_length)

	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()

files = ["train_0", "train_1", "valid_1", "valid_0"]#, "test_1", "test_0"]
#number = [30, 30, 10, 10, 5, 5]  # number of images
cnt = [101, 95, 34, 32, 19, 16]

x = 0
for folder in files:
	print(folder)
	x += 1
	f = open("expt_2/"+folder+".txt",'r')
	lines = f.readlines()
	#src = "../deepak/DB_HnE_101_anno_cent/label_"+folder.split('_')[1]+'/'
	src = "./expt_4/PetImageslabel_"+folder.split('_')[1]+'/'
	dest = './expt_4/data_dump/'+folder.split('_')[0]+"/label_"+folder.split('_')[1]
	if not os.path.exists(dest):
		os.mkdir(dest)
		
	p = 0
	for index in range(len(lines)):
		p += 1
		imgs = os.listdir(src+lines[index].split('\n')[0]+'_label_'+folder.split('_')[1]+'/')
		if( len(imgs) > 0):
			os.system("cp "+src+lines[index].split('\n')[0]+'_label_'+folder.split('_')[1]+"/"+lines[index].split('\n')[0]+'* '+dest)
		print_progress(p, cnt[x-1])

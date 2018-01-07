# -*- coding: utf-8 -*-
import sys
import os
import shutil

from os.path import join

# Global Vairables
dest_dump='./'
src_dir='/home/Drive2/deepak/DB_HnE_299_anno_cent/'

n_classes=2
labels_list=[3, 45]


def main():

	files = ["train_3", "train_45", "valid_3", "valid_45"]#, "test_1", "test_0"]
	cnt = [101, 95, 34, 32, 19, 16]

	x = 0
	for folder in files:
		print(folder)
		x += 1
		f = open(folder+".txt",'r')
		lines = f.readlines()

		dest_split_dir = join(dest_dump, folder.split('_')[0])
		if not os.path.exists(dest_split_dir):
			make_dir(dest_split_dir)

		dest_label_dir = join(dest_split_dir, "label_"+folder.split('_')[1])

		p = 0
		for line in lines:
			p += 1
			# print line
			patient_path = line.split("\n")[0]
			src_patient_dir = '/home/Drive2/deepak/DB_HnE_299_anno_cent'+patient_path
			# print src_patient_dir
			imgs = os.listdir(src_patient_dir)
			if( len(imgs) > 0):
				os.system("cp "+src_patient_dir+'/* '+dest_label_dir)

			print_progress(p, cnt[x-1])

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

# Makes directory
def make_dir(dir, needlabel=True):
	if os.path.exists(dir):
		shutil.rmtree(dir)
	os.mkdir(dir)
	if needlabel is True:
		for label in labels_list:
			os.mkdir(os.path.join(dir, "label_"+str(label)))

if __name__ == '__main__':
	main()

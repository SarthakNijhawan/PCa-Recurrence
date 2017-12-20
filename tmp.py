import os
import sys
import shutil

src = sys.argv[1]
dest = sys.argv[2]
number = sys.argv[3]

files_list = os.listdir(src)

for i in range(number):
	src_path = os.path.join(src, files_list[i])
	dest_path = os.path.join(dest, files_list[i])
	shutil.copytree(src_path, dest_path)

print("Completed !!")

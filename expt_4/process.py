import os

files = ["train_0", "train_1", "valid_0", "valid_1"]
num = [10000, 10000, 2500, 2500]

count = [0, 0]

for index in range(4):
	fid = open(files[index]+".txt", "w")
	label = int(files[index].split("_")[-1])
	init_num = count[label]

	for i in range(init_num, init_num+num[index], 1):
		fid.write(str(i)+"\n")

	fid.close()

	count[label] += num[index]

import sys
import pickle
import cv2
import os 
import numpy as np

#creates 224 and 101


picklePath = "../outannoCentPickles"
DB_H_path = "../sorted_patient_wise_h_dataset"
DB_E_path = "../sorted_patient_wise_e_dataset"
DB_HnE_path = "../sorted_patient_wise_normalized_dataset"

dest_path = ["../DB_HnE_101_outanno_cent","../DB_H_101_outanno_cent","../DB_E_101_outanno_cent"]

for folder in dest_path:
	try:
		os.mkdir("%s"%(folder))
		os.mkdir("%s/label_0"%(folder))
		os.mkdir("%s/label_1"%(folder))
	except:
		print("%s folder exist sir!"%(folder))


labels = ["label_0","label_1"]
#print("LOl")
#f_written = open("writtenData.txt",'r')
#writtenPatients = f_written.readlines()
#writtenPatients = [x.split('\n')[0] for x in writtenPatients]

#f_299 = open("segment299Log_nilgiri.txt",'w')
f_101 = open("segment101Log_nilgiri.txt",'w')

for label in labels:
	p = 0
	for patient in os.listdir(picklePath+'/'+label):
		print patient
		p += 1
		for folder in dest_path:
			print folder,label,patient
			try:
				os.mkdir("%s/%s/%s"%(folder,label,patient))		
			except:
				print "exists"
		pick = 0
		print(p)
		for pickleFile in os.listdir(picklePath+'/'+label+'/'+patient):	
			centreList = []			
			pick += 1	
			data = open(picklePath+'/'+label+'/'+patient+'/'+pickleFile,"rb")
			centreList = pickle.load(data)
			data.close()
		
			# f_299.write(label+","+patient.split('_')[0]+","+ str(p)+ "/"+str(len(os.listdir(picklePath+'/'+label)))+","+str(pick)+"/"+str(len(os.listdir(picklePath+'/'+label+'/'+patient)))+",writtenCount,")
			f_101.write(label+","+patient.split('_')[0]+","+ str(p)+ "/"+str(len(os.listdir(picklePath+'/'+label)))+","+str(pick)+"/"+str(len(os.listdir(picklePath+'/'+label+'/'+patient)))+",writtenCount,")
			img = pickleFile.split('.')[0]			
			H_img = cv2.imread(DB_H_path+'/'+label+'/'+patient+'/'+img+".tif")
			# print H_img.shape
			E_img = cv2.imread(DB_E_path+'/'+label+'/'+patient+'/'+img+".tif")
			# print E_img.shape
			# print DB_HnE_path+'/'+label+'/'+patient+'/'+img+".tif"
			HnE_img = cv2.imread(DB_HnE_path+'/'+label+'/'+patient+'/'+img+".tif")
			# print HnE_img.shape
			# count_299 = 0
			count_101 = 0
			for centre in centreList:
				#count += 1
				x = centre[0]
				y = centre[1]
				# if(x>=150 and y>=150 and x<=1850 and y<=1850):
				# 	patch=HnE_img[y-150:y+150,x-149:x+149,:]
				# 	patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
				# 	patch_pass=np.sum(patch<0.7*255)/62580.0
				# 	if(patch_pass>1.0):
				# 		count_299 += 1
				# 		# cv2.imwrite(dest_path[0]+'/'+label+'/'+patient+'/'+patient.split('_')[0]+'_'+str(x)+'_'+str(y)+'_'+img+".png",E_img[y-112:y+112,x-112:x+112,:])
				# 		# cv2.imwrite(dest_path[1]+'/'+label+'/'+patient+'/'+patient.split('_')[0]+'_'+str(x)+'_'+str(y)+'_'+img+".png",H_img[y-112:y+112,x-112:x+112,:])
				# 		cv2.imwrite(dest_path[0]+'/'+label+'/'+patient+'/'+patient.split('_')[0]+'_'+str(x)+'_'+str(y)+'_'+img+".png",HnE_img[y-150:y+150,x-149:x+149,:])
					##optional here this is for 101
				if(x>=50 and y>=50 and x<=1949 and y<=1949):
					count_101 += 1
					# print dest_path[2]+'/'+label+'/'+patient+'/'+patient.split('_')[0]+'_'+str(x)+'_'+str(y)+'_'+img+".png"
					cv2.imwrite(dest_path[2]+'/'+label+'/'+patient+'/'+patient.split('_')[0]+'_'+str(x)+'_'+str(y)+'_'+img+".png",E_img[y-50:y+51,x-50:x+51,:])
					cv2.imwrite(dest_path[1]+'/'+label+'/'+patient+'/'+patient.split('_')[0]+'_'+str(x)+'_'+str(y)+'_'+img+".png",H_img[y-50:y+51,x-50:x+51,:])					
					cv2.imwrite(dest_path[0]+'/'+label+'/'+patient+'/'+patient.split('_')[0]+'_'+str(x)+'_'+str(y)+'_'+img+".png",HnE_img[y-50:y+51,x-50:x+51,:])
					# print dest_path[2]+'/'+label+'/'+patient+'/'+patient.split('_')[0]+'_'+str(x)+'_'+str(y)+'_'+img+".png"
					
			
			# f_299.write(str(count_299)+'/'+str(len(centreList))+'\n')
			f_101.write(str(count_101)+'/'+str(len(centreList))+'\n')
					
# f_299.close()
f_101.close()				

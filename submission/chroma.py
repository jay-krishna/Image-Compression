import numpy as np
import cv2
import sys
import math
import os
import shutil
from sewar.full_ref import psnr,mse,rmse
import matplotlib.pyplot as plt

arr=[[],[],[],[],[],[]]

def FetchSteps(sample):
	if (sample[1]==1):
		if (sample[2]==0):
			return [4,2]
		else:
			return [4,1]
	elif (sample[1]==2):
		if (sample[2]==0):
			return [2,2]
		else:
			return [2,1]
	else:
		if(sample[2]==0):
			return [1,2]
		else:
			return [1,1]

def repeat(array,steps):
	x=np.stack([array for _ in range(steps[0])], axis=0)
	y=np.stack([x for _ in range(steps[1])], axis=1)

	return y

def prepare():
	if os.path.exists('output'):
		shutil.rmtree('output/')
	os.makedirs('output')

def main(sample):
	input_file_name=sys.argv[1]
	original_file_object=cv2.imread(input_file_name)
	original_file_size=original_file_object.shape[0]*original_file_object.shape[1]*original_file_object.shape[2]

	converted_image=cv2.cvtColor(original_file_object,cv2.COLOR_BGR2YCR_CB)
	y,cr_original,cb_original = cv2.split(converted_image)

	steps=FetchSteps(sample)

	
	# cr_new=np.zeros((a,b))
	
	cr_new=cr_original[::steps[0],::steps[1]]
	cb_new=cb_original[::steps[0],::steps[1]]

	# for 

	compressed_file_size=y.shape[0]*y.shape[1]+cb_new.shape[0]*cb_new.shape[1]+cr_new.shape[0]*cr_new.shape[1]
	
	cr_de=np.repeat(cr_new,steps[0],axis=0)
	cr_de=np.repeat(cr_de,steps[1],axis=1)

	cb_de=np.repeat(cb_new,steps[0],axis=0)
	cb_de=np.repeat(cb_de,steps[1],axis=1)

	while(y.shape[0] != cr_de.shape[0]):
		cr_de=np.delete(cr_de,cr_de.shape[0]-1,0)

	while(y.shape[0] != cb_de.shape[0]):
		cb_de=np.delete(cb_de,cb_de.shape[0]-1,0)

	while(cb_original.shape[1]!=cb_de.shape[1]):
		cb_de=np.delete(cb_de,cb_de.shape[0]-1,1)

	while(cr_original.shape[1]!=cr_de.shape[1]):
		cr_de=np.delete(cr_de,cr_de.shape[0]-1,1)	

	# print(y.shape)
	# print(cb_original.shape)
	# print(cr_original.shape)
	# print(cb_de.shape)
	# print(cr_de.shape)
	
	combined = np.dstack((y, cr_de,cb_de))
	decoded = cv2.cvtColor(combined,cv2.COLOR_YCR_CB2BGR)
	difference=np.dstack((y-y,cr_original-cr_de,cb_original-cb_de))

	# file1=input_file_name.split[0]
	# print(type(input_file_name))
	file2=input_file_name.split('.')[1]
	a=sample[0]
	b=sample[1]
	c=sample[2]
	add=str(a)+str(b)+str(c)


	cv2.imwrite('./output/decoded'+add+'.'+file2,decoded)
	cv2.imwrite('./output/decoded_diff_'+add+'.'+file2,difference)
	ratio=(original_file_size)/compressed_file_size
	print("Compression Ratio: "+str(ratio))
	# print("psnr")
	print("PSNR: "+str(psnr(combined,converted_image)))
	print("MSE: "+str(mse(combined,converted_image)))
	print("RMSE: "+str(rmse(combined,converted_image)))
	# psnrl.append(psnr(combined,converted_image))
	# rmsl.append(rmse(combined,converted_image))
	# com.append(ratio)
	# arr[counter].append(rmse(combined,converted_image))
	# print(msssim(original_file_object,decoded))

def __init__():
	'''
		4,1,0
		4,1,1
		4,2,0
		4,2,2
		4,4,0
		4,4,4
	'''
	# file_list=['/home/singular/img/lena.png','/home/singular/img/peppers.png','/home/singular/img/1.jpg','/home/singular/img/3.jpg','/home/singular/img/4.jpg']

	# file_list_name=['Lena','Peppers','Mountain','Water','Street']

	prepare()
	options=[(4,1,0),(4,1,1),(4,2,0),(4,2,2),(4,4,0),(4,4,4)]
	# x=np.arange(1,7,1)
	# counter=0
	# for file in file_list:
	for o in options:
		print("*****************")
		print("Sampling Rate: "+str(o))
		main(o)

	

	# # plt.subplot(311)
	# 	plt.plot(x,arr[counter],'o-',label=str(file_list_name[counter]))
	# 	plt.grid(True)
	# # plt.show()


	# # plt.subplot(312)
	# # plt.plot(x,rmsl,'^-',label="RMSE")
	# # plt.grid(True)
	# # plt.title("Chroma Subsampling")
	# 	plt.legend(loc='upper right')
	# 	plt.title("Chroma Subsampling(RMSE)")
	# 	counter=counter+1

	# 	# plt.ylabel("db")

	# plt.savefig("2.png")
	# plt.show()


	# plt.plot(x,psnrl,'s-',color='r',label="PSNR")
	# plt.grid(True)
	# plt.ylabel("db")
	# plt.title("Chroma Subsampling")
	# plt.legend(loc='upper left')
	# plt.savefig("2.png")
	# plt.show()

__init__()
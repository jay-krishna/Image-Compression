import numpy as np
import cv2
import sys
import math
import os
import shutil
import matplotlib.pyplot as plt
from sewar.full_ref import psnr,mse,rmse

def prepare():
	if os.path.exists('output'):
		shutil.rmtree('output/')
	os.makedirs('output')

# arr=[[],[],[],[],[],[]]
ratios=[]


def main():

	prepare()

	original_file_name=sys.argv[1]
	original_file_object=cv2.imread(original_file_name)
	original_file_object=cv2.cvtColor(original_file_object, cv2.COLOR_BGR2RGB)
	original_file_size=original_file_object.shape[0]*original_file_object.shape[1]*original_file_object.shape[2]
	# original_file_object=cv2.resize(original_file_object,(512, 512))

	r,g,b=cv2.split(original_file_object)
	# original_file_object.clear()

	final_r=[]
	final_b=[]
	final_g=[]
	count=0
	
	for i in range(0,r.shape[0]):

		temp_r=[]
		temp_b=[]
		temp_g=[]
		count_r=0
		count_b=0
		count_g=0
		
		for j in range(0,r.shape[1]-1):
			r_=r[i][j]
			r__=r[i][j+1]

			# print(str(r_)+" "+str(r__))

			if r_==r__:
				count_r=count_r+1
			else:
				if count_r>0:
					temp_r.append((r_,count_r+1))
					# temp_r.append(_)
					count_r=0
					# count=count+3
				else:
					count_r=0
					temp_r.append(r_)
					# count=count+1

			b_=b[i][j]
			b__=b[i][j+1]

			if b_==b__:
				count_b=count_b+1
			else:
				if count_b>0:
					temp_b.append((b_,count_b+1))
					# temp_r.append(_)
					count_b=0
					# count=count+3
				else:
					count_b=0
					temp_b.append(b_)
					# count=count+1

			g_=g[i][j]
			g__=g[i][j+1]

			if g_==g__:
				count_g=count_g+1
			else:
				if count_g>0:
					temp_g.append((g_,count_g+1))
					# temp_r.append(_)
					count_g=0
					# count=count+3
				else:
					count_g=0
					temp_g.append(g_)
					# count=count+1

		if count_r>0:
			temp_r.append((r_,count_r+1))
		else:
			temp_r.append(r__)

		if count_b>0:
			temp_b.append((b_,count_b+1))
		else:
			temp_b.append(b__)

		if count_g>0:
			temp_g.append((g_,count_g+1))
		else:
			temp_g.append(g__)

		final_r.append(temp_r)
		final_b.append(temp_b)
		final_g.append(temp_g)

		count=count+len(temp_r)+len(temp_b)+len(temp_g)


	# print(r)
	# print(final_r)
	# final_size=len(final_r)+len(final_g)+len(final_b)
	# print(final_size)
	# print(original_file_size)
	ratio=original_file_size/count
	# ratios.append(ratio)


	recon_r=[]
	recon_g=[]
	recon_b=[]

	for x in final_r:
		ttt=[]
		for y in x:
			if(isinstance(y,tuple)):
				for i in range(0,y[1]):
					ttt.append(y[0])
			else:
				ttt.append(y)

		recon_r.append(ttt)

	for x in final_g:
		ttt=[]
		for y in x:
			if(isinstance(y,tuple)):
				for i in range(0,y[1]):
					ttt.append(y[0])
			else:
				ttt.append(y)

		recon_g.append(ttt)

	for x in final_b:
		ttt=[]
		for y in x:
			if(isinstance(y,tuple)):
				for i in range(0,y[1]):
					ttt.append(y[0])
			else:
				ttt.append(y)

		recon_b.append(ttt)

	# print(len(recon_r[1]))

	nrecon_r=np.array(recon_r)
	nrecon_b=np.array(recon_b)
	nrecon_g=np.array(recon_g)

	# print(nrecon_r.shape)
	# print(nrecon_b.shape)
	# print(nrecon_g.shape)

	file2=original_file_name.split('.')[1]


	combined=np.dstack((nrecon_r,nrecon_g,nrecon_b))
	difference=np.dstack((r-nrecon_r,g-nrecon_g,b-nrecon_b))
	combined=cv2.cvtColor(combined,cv2.COLOR_RGB2BGR)
	difference=cv2.cvtColor(difference,cv2.COLOR_RGB2BGR)
	cv2.imwrite('./output/decoded'+'.'+file2,combined)
	cv2.imwrite('./output/decoded_diff_'+'.'+file2,difference)
	original_file_object=cv2.cvtColor(original_file_object, cv2.COLOR_RGB2BGR)

	print("Ratio")
	print(ratio)
	print("MSE")
	print(mse(original_file_object,combined))
	print("psnr")
	print(psnr(original_file_object,combined))


def __init__():
	# file_list=['/home/singular/img/lena.png','/home/singular/img/peppers.png','/home/singular/img/1.jpg','/home/singular/img/3.jpg','/home/singular/img/4.jpg']

	# file_list_name=['Lena','Peppers','Mountain','Water','Street']
	# # file_list=['/home/singular/Desktop/1_1.jpeg']
	# # file_list_name=['Random']
	# counter=0

	# for file in file_list:
	main()
		# counter=counter+1

	# x=np.arange(1,6,1)
	# plt.grid(True,axis='y')

	# plt.bar(x, ratios, align='center')
	# # plt.plot(x,ratios,'o-')
	# plt.xticks(x, file_list_name)
	# plt.title("Run Length Encoding(Ratio)")
	# plt.show()

	# print(ratios)

__init__()
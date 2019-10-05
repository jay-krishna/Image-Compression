import numpy as np
import cv2
import sys
import math
import os
import shutil

def prepare():
	if os.path.exists('output'):
		shutil.rmtree('output/')
	os.makedirs('output')


def main():

	prepare()

	original_file_name=sys.argv[1]
	original_file_object=cv2.imread(original_file_name)
	original_file_object=cv2.cvtColor(original_file_object, cv2.COLOR_BGR2RGB)
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

	# print(count)

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

	combined=np.dstack((nrecon_r,nrecon_g,nrecon_b))
	difference=np.dstack((r-nrecon_r,g-nrecon_g,b-nrecon_b))
	# combined=np.resize(combined,(combined.shape[1],combined.shape[2]))
	print(combined.shape)
	combined=cv2.cvtColor(combined,cv2.COLOR_RGB2BGR)
	cv2.imwrite('/home/singular/img/15.jpg',combined)
	cv2.imwrite('/home/singular/img/16.jpg',difference)
	original_file_object=cv2.cvtColor(original_file_object, cv2.COLOR_RGB2BGR)
	# cv2.imwrite('/home/singular/img/16.jpg',original_file_object)


	# print(r- nrecon_r)
	# print(np.max(r- nrecon_r))
	# print(np.min(r- nrecon_r))
	# # np.savetxt('data.txt',r-nrecon_r)
	# # print(nrecon_r)

	# # print(b- recon_b)
	# print(np.max(b- nrecon_b))
	# print(np.min(b- nrecon_b))
	# # np.savetxt('data1.txt',b-nrecon_b)
	# # print(recon_b)

	# # print(g- recon_g)
	# print(np.max(g- nrecon_g))
	# print(np.min(g- nrecon_g))
	# np.savetxt('data2.txt',g-nrecon_g)
	# print(recon_g)

	print("MSE")
	print(mse(original_file_object,combined))
	print("psnr")
	print(psnr(original_file_object,combined))


def __init__():
	main()

__init__()
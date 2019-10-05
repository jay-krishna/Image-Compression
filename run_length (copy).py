import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
import math



def main():
	original_file_name=sys.argv[1]
	original_file_object=cv2.imread(original_file_name)
	original_file_object=cv2.cvtColor(original_file_object, cv2.COLOR_BGR2RGB)
	original_file_object=cv2.resize(original_file_object,(256, 256))

	r,g,b=cv2.split(original_file_object)
	count=0
	count_all=0
	count_r=0
	count_g=0
	count_b=0
	final_lst_r=[]
	final_lst_g=[]
	final_lst_b=[]

	for x in range(0,r.shape[0]):
		lst_r=[]
		lst_b=[]
		lst_g=[]
		# prev_r=-1
		# prev_b=-1
		# prev_g=-1

		for y in range(1,r.shape[1]):
			# t=(r[x][y],g[x][y],b[x][y])
			# t_=(r[x][y-1],g[x][y-1],b[x][y-1])
			count_all=count_all+1

			t=r[x][y]
			t_=r[x][y-1]
			# (abs(int(t)-int(t_))<5):
			if(t==t_):
				count_r=count_r+1
			else:
				if count_r>0:
					lst_r.append((t_,count_r))
					count_r=0
				else:
					lst_r.append(t)
					count_r=0

			t=g[x][y]
			t_=g[x][y-1]
			if(t==t_):
				count_g=count_g+1
				# count=count+1
			else:
				if count_g>0:
					lst_g.append((t_,count_g))
					count_g=0
				else:
					lst_g.append(t)
					count_g=0

			t=b[x][y]
			t_=b[x][y-1]
			if(t==t_):
				count_b=count_b+1
				# count=count+1
			else:
				if count_b>0:
					lst_r.append((t_,count_b))
					count_b=0
				else:
					lst_b.append(t)
					count_b=0

		count=count+len(lst_b)+len(lst_g)+len(lst_r)

		final_lst_r.append(lst_r)
		final_lst_b.append(lst_b)
		final_lst_g.append(lst_g)


	# print(len(unique.keys()))
	print((float(3*count_all-count)/float(3*count_all))*100)
	print(len(final_lst_r[0]))

	decoded_r=[]
	decoded_g=[]
	decoded_b=[]

	for _ in final_lst_r:
		temp=[]
		# print(_)
		for __ in _:
			# print(__)
			if(isinstance(__,tuple)):
				for t in range(0,__[1]):
					temp.append(__[0])
			else:
				temp.append(__)
		print(len(temp))
		decoded_r.append(temp)

	for _ in final_lst_g:
		temp=[]
		for __ in _:
			if(isinstance(__,tuple)):
				for t in range(0,__[1]):
					temp.append(__[0])
			else:
				temp.append(__)
		decoded_g.append(temp)

	for _ in final_lst_b:
		temp=[]
		for __ in _:
			if(isinstance(__,tuple)):
				for t in range(0,__[1]):
					temp.append(__[0])
			else:
				temp.append(__)

		decoded_b.append(temp)


	# print(len(decoded_b))
	# print(len(decoded_g))
	# print(len(decoded_r[0]))

	# np_r=np.array(decoded_r)
	# np_g=np.array(decoded_g)
	# np_b=np.array(decoded_b)

	# print(np_r.shape)
	# print(np_g.shape)
	# print(np_b.shape)

	# combined=np.dstack((np_r,np_g,np_b))
	# combined=np.resize(combined,(combined.shape[1],combined.shape[2]))
	# print(combined.shape)
	# combined=cv2.cvtColor(combined,cv2.COLOR_RGB2BGR)
	# cv2.imwrite('/home/singular/img/15.jpg',combined)


def __init__():
	main()

__init__()
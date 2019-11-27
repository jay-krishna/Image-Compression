import numpy as np
import cv2
import sys
import math
import os
import shutil
from sewar.full_ref import psnr,mse,rmse
from matplotlib import pyplot as plt
# from zigzagf import zigzag
from heapq import heappush, heappop, heapify
from collections import defaultdict,Counter

def zigzag(input):
	h = 0
	v = 0
	vmin = 0
	hmin = 0
	vmax = input.shape[0]
	hmax = input.shape[1]
	i = 0
	output = np.zeros(( vmax * hmax))

	while ((v < vmax) and (h < hmax)):
		temp = (h+v)%2
		if (temp!= 0):
			if ((v == vmax -1) and (h <= hmax -1)):       
				output[i] = input[v, h] 
				h = h + 1
				i = i + 1
			elif ((v < vmax -1) and (h > hmin)):       
				output[i] = input[v, h] 
				v = v + 1
				h = h - 1
				i = i + 1

			elif (h == hmin):                
				output[i] = input[v, h] 
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
					i = i + 1          
		else:                                   
			if (v == vmin):
				output[i] = input[v, h]       
				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        
					i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):        
				output[i] = input[v, h] 
				v = v - 1
				h = h + 1
				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):
				output[i] = input[v, h] 
				v = v + 1
				i = i + 1

		if ((v == vmax-1) and (h == hmax-1)):         

			output[i] = input[v, h] 
			break

	return output

def showImage(img):
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    plt.xticks([]),plt.yticks([])
    plt.show()

def HuffmanEncoder(symb2freq):
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def HuffmanKernel(vector):
	symb2freq_r = defaultdict(int)
	# huff_r = HuffmanEncoder(symb2freq_r)

	for _ in vector[0]:
		# print(_)
		if(isinstance(_, tuple)):
			symb2freq_r[_[0]]+=_[1]
			symb2freq_r[_[1]]+=1
		else:
			symb2freq_r[_]+=1

	# print(symb2freq_r)
	huff_r = HuffmanEncoder(symb2freq_r)
	dic_r={}
	for p in huff_r:
		dic_r[p[0]]=p[1]
		# print(p)
		# print ("%s\t%s\t%s\t\t%s\t\t%s" % (p[0],p[1],symb2freq_r[p[0]],symb2freq_r[p[0]]/(ll.shape[0]*ll.shape[1]),len(p[1])))
	    # print("%s\t%s\t%s"%(p[0],symb2freq_r[p[0]],symb2freq_r[p[0]]/(ll.shape[0]*ll.shape[1])))

	total=0
	# for x in ll:
	# 	for y in x:
	# 		total+=len(dic_r[y])

	# print(total)

	for _ in vector[0]:
		if(isinstance(_,tuple)):
			# print(_,end=" ")
			# print(dic_r[_[0]],end=" ")
			# print(dic_r[_[1]],end=" ")

			total+=len(dic_r[_[0]])
			total+=len(dic_r[_[1]])
			# print(total,end='\n')
		else:
			# print(_,end=" ")
			# print(dic_r[_],end=" ")

			total+=len(dic_r[_])
			# print(total,end='\n')

	# print(total)

	return total

def runs(r):
	count_r=0
	final_r=[]
	count=0

	for i in range(r.shape[0]-1):
		r_=r[i]
		r__=r[i+1]

		# print(str(r_)+" "+str(r__))

		if r_==r__:
			count_r=count_r+1
		else:
			if count_r>0:
				final_r.append((r_,count_r+1))
				count+=2
				# temp_r.append(_)
				count_r=0
				# count=count+3
			else:
				count_r=0
				final_r.append(r_)
				count+=1
				# count=count+1

	if count_r>0:
		final_r.append((r_,count_r+1))
		count+=2
	else:
		final_r.append(r__)
		count+=1

	return (final_r,count)

def prepare():
	if os.path.exists('output'):
		shutil.rmtree('output/')
	os.makedirs('output')

def transform(block,k,iden):
	# print(block)
	block=block-128
	block_dct=cv2.dct(block/255.0)*255
	# print(block_dct.astype(int))

	if iden==0:
		block_dct=block_dct/(ll*k)

	if iden==1:
		block_dct=block_dct/(lc*k)

	# print(block_dct)

	return block_dct

def itransform(block,k,iden):
	if iden==0:
		# print("0")
		block=block*ll*k

	if iden==1:
		# print("1")
		block=block*lc*k

	block_dct=cv2.idct(block/255.0)*255
	block_dct=block_dct-128
	# print(block_dct)
	return block_dct

def filewrite(matrix):
	f= open("image.txt",'w')
	
	for i in range(0,matrix.shape[0]):
		for j in range(0,matrix.shape[1]):
			f.write(str(matrix[i][j]))
			f.write(" ")
		f.write('\n')

	f.close()	
	# stream=[]
	# while i in matrix.shape[0]:
	# 	if(matrix[i]!=0):
	# 		stream.append()


def main():
	k=10

	if(k<=50):
		k=50/k
	elif k==100:
		k=2-(99/50)
	else:
		k=2-(k/50)

	original_file_name=sys.argv[1]
	original_file_object=cv2.imread(original_file_name)
	original_file_object=cv2.cvtColor(original_file_object, cv2.COLOR_BGR2YCR_CB)
	
	x=original_file_object.shape[0]
	y=original_file_object.shape[1]
	if(x>8):
		x=int(x/8)
		x=x*8
	if(y>8):
		y=int(y/8)
		y=y*8

	original_file_object=cv2.resize(original_file_object,(y,x))

	y,cr,cb=cv2.split(original_file_object)

	y_dct=np.zeros(y.shape,dtype=np.int)
	cr_dct=np.zeros(cr.shape,dtype=np.int)
	cb_dct=np.zeros(cb.shape,dtype=np.int)
	y_idct=np.zeros(y.shape,dtype=np.uint8)
	cr_idct=np.zeros(cr.shape,dtype=np.uint8)
	cb_idct=np.zeros(cb.shape,dtype=np.uint8)
	# print(cr_new.shape)

	for i in range(0,cb.shape[0],8):
		for j in range(0,cb.shape[1],8):
			# print(str(i)+" "+str(j))

			# if(cb.shape[0]-i<0 or cb.shape[1]-j<0)
				# break

			cr_dct[i:i+8,j:j+8]=transform(cr[i:i+8,j:j+8],k,1)
			cb_dct[i:i+8,j:j+8]=transform(cb[i:i+8,j:j+8],k,1)
			y_dct[i:i+8,j:j+8]=transform(y[i:i+8,j:j+8],k,0)

			# print(y[i:i+8,j:j+8])
			# print(y_dct[i:i+8,j:j+8])
			# print("yes")

			cr_idct[i:i+8,j:j+8]=itransform(cr_dct[i:i+8,j:j+8],k,1)
			cb_idct[i:i+8,j:j+8]=itransform(cb_dct[i:i+8,j:j+8],k,1)
			y_idct[i:i+8,j:j+8]=itransform(y_dct[i:i+8,j:j+8],k,0)

	# print(cb_dct)

	# for i in range(0,cb.shape[0],8):
	# 	for j in range(0,cb.shape[1],8):
	# 		# print(str(i)+" "+str(j))
	# 		cr_idct[i:i+8,j:j+8]=itransform(cr_dct[i:i+8,j:j+8],k,1)
	# 		cb_idct[i:i+8,j:j+8]=itransform(cb_dct[i:i+8,j:j+8],k,1)
	# 		y_idct[i:i+8,j:j+8]=itransform(y_dct[i:i+8,j:j+8],k,0)

			# print(cr_idct[i:i+8,j:j+8])
			# print(cr[i:i+8,j:j+8]-cr_idct[i:i+8,j:j+8])
	# print(y)
	# print(y-y_idct)
	# print(cr)
	# print(cr-cr_idct)
	# print(cb)
	# print(cb_idct)
	# co

	# print(y)
	# print(y_idct)
	# print(cr)
	# print(cr_idct)
	# print(cb)
	# print(cb_idct)

	# print(y_dct)
	# print(zigzag(y_dct))
	# print(runs(zigzag(y_dct)))
	# print(y.shape[0]*y.shape[1])
	# print(len(runs(zigzag(y_dct))))

	# print(cr_dct)
	# print(zigzag(cr_dct))
	# print(runs(zigzag(cr_dct)))
	# print(len(runs(zigzag(cr_dct))))

	# print(cb_dct)
	# print(zigzag(cb_dct))
	# print(runs(zigzag(cb_dct)))
	# print(len(runs(zigzag(cb_dct))))

	original_size=y.shape[0]*y.shape[1]*3*8
	run_y=runs(zigzag(y_dct))
	run_cb=runs(zigzag(cb_dct))
	run_cr=runs(zigzag(cr_dct))

	total_y=HuffmanKernel(run_y)
	total_cb=HuffmanKernel(run_cb)
	total_cr=HuffmanKernel(run_cr)

	new_file=total_y+total_cb+total_cr
	# print("After Run Length: ",end="")
	# print((original_size/8)/(run_y[1]+run_cb[1]+run_cr[1]))
	# print("After Huffman: ",end="")
	# print(original_size/new_file)
	# print(original_size)
	# print(new_file)


	combined = np.dstack((y_idct, cr_idct,cb_idct))
	decoded = cv2.cvtColor(combined,cv2.COLOR_YCR_CB2BGR)
	# decoded=combined
	# print(original_file_object.shape)
	# print(decoded.shape)

	print("Compression Ratio after runlength: "+str((original_size/8)/(run_y[1]+run_cb[1]+run_cr[1])))
	print("Compression Ratio after huffman: "+str(original_size/new_file))
	print("PSNR: "+str(psnr(combined,original_file_object)))
	print("MSE: "+str(mse(combined,original_file_object)))
	print("RMSE: "+str(rmse(combined,original_file_object)))

	# f=open("./output/res.txt",'w')
	# f.write(str((original_size/8)/(run_y[1]+run_cb[1]+run_cr[1])))
	# f.write('\n')
	# f.write(str(original_size/new_file))
	# f.write('\n')
	# f.write(str(psnr(combined,original_file_object)))
	# f.write('\n')
	# f.write(str(mse(combined,original_file_object)))
	# f.write('\n')
	# f.write(str(rmse(combined,original_file_object)))
	# f.close()

	file2=original_file_name.split('.')[1]
	cv2.imwrite('./output/decoded'+'.'+file2,decoded)
	cv2.imwrite('./output/diff'+'.'+file2,original_file_object-combined)

	# filewrite(y_dct)
	# print(y_dct)

	# print(y_idct.shape)
	# print(y.shape)

	# print(cr_idct.shape)
	# print(cr.shape)

	# print(cb_idct.shape)
	# print(cb.shape)

	# print(original_file_object.shape)
	# print(combined.shape)
	# print(decoded.shape)

	# showImage(combined)
	# showImage(original_file_object)

def __init__():
	prepare()
	main()

# ll=np.array([[16,11,10,16,24,40,51,61],
# [12,12,14,19,26,58,60,55],
# [14,13,16,24,40,57,69,56],
# [14,17,22,29,51,87,80,62],
# [18,22,37,56,68,109,103,77],
# [24,35,55,64,81,104,113,92],
# [9,64,78,87,103,121,120,101],
# [72,92,95,98,112,100,103,99]])

# lc=np.array([[17,18,24,47,99,99,99,99],
# [18,21,26,66,99,99,99,99],
# [24,26,56,99,99,99,99,99],
# [47,66,99,99,99,99,99,99],
# [99,99,99,99,99,99,99,99],
# [99,99,99,99,99,99,99,99],
# [99,99,99,99,99,99,99,99],
# [99,99,99,99,99,99,99,99]])

# ll=np.array([[3,2,2,3,5,8,10,12],
# 	        [2,2,3,4,5,12,12,11],
# 	        [3,3,3,5,8,11,14,11],
# 	        [3,3,4,6,10,17,16,12],
# 	        [4,4,7,11,14,22,21,15],
# 	        [5,7,11,13,16,12,23,18],
# 	        [10,13,16,17,21,24,24,21],
# 	        [14,18,19,20,22,20,20,20]])

ll=np.array([[3,5,7,9,11,13,15,17],
	[5,7,9,11,13,15,17,19],
	[7,9,11,13,15,17,19,21],
	[9,11,13,15,17,19,21,23],
	[11,13,15,17,19,21,23,25],
	[13,15,17,19,21,23,25,27],
	[15,17,19,21,23,25,27,29],
	[17,19,21,23,25,27,28,31]])

# ll = np.array([[80,60,50,80,120,200,255,255],
#                 [55,60,70,95,130,255,255,255],
#                 [70,65,80,120,200,255,255,255],
#                 [70,85,110,145,255,255,255,255],
#                 [90,110,185,255,255,255,255,255],
#                 [120,175,255,255,255,255,255,255],
#                 [245,255,255,255,255,255,255,255],
#                 [255,255,255,255,255,255,255,255]])

# ll = np.array([[16,11,10,16,24,40,51,61],
#                 [12,12,14,19,26,58,60,55],
#                 [14,13,16,24,40,57,69,56],
#                 [14,17,22,29,51,87,80,62],
#                 [18,22,37,56,68,109,103,77],
#                 [24,35,55,64,81,104,113,92],
#                 [49,64,78,87,103,121,120,101],
#                 [72,92,95,98,112,100,130,99]])

lc=ll

__init__()
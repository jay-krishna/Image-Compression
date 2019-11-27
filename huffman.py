from heapq import heappush, heappop, heapify
from collections import defaultdict,Counter
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import math
import os
import shutil
from sewar.full_ref import psnr,mse,rmse

arr=[]

def prepare():
    if os.path.exists('output'):
        shutil.rmtree('output/')
    os.makedirs('output')
 
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

def main():
	
	prepare()
	file=sys.argv[1]
	file=cv2.imread(file)
	file=cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
	original_file_size=file.shape[0]*file.shape[1]*file.shape[2]

	r,g,b=cv2.split(file)

	symb2freq_r = defaultdict(int)
	symb2freq_g = defaultdict(int)
	symb2freq_b = defaultdict(int)
	dic_r={}
	dic_g={}
	dic_b={}
	dic_r_rev={}
	dic_g_rev={}
	dic_b_rev={}
	# np_r=np.zeros(256)
	# np_b=np.zeros(256)
	# np_g=np.zeros(256)

	for x in r:
	    for y in x:
	        symb2freq_r[y] += 1

	huff_r = HuffmanEncoder(symb2freq_r)
	# print(r)
	exp_len_r=0
	# entropy_r=0
	# print("Symbol\tFrequency\tProbability\tHuffman Code")
	for p in huff_r:
	    # print ("%s\t%s\t\t%s\t\t%s" % (p[0], symb2freq_r[p[0]],symb2freq_r[p[0]]/(r.shape[0]*r.shape[1]),len(p[1])))
	    dic_r[p[0]]=p[1]
	    dic_r_rev[p[1]]=p[0]

	    # prob=(symb2freq_r[p[0]]/(r.shape[0]*r.shape[1]))
	    # exp_len_r=exp_len_r+prob*len(p[1])
	    # prob_inv=1/prob
	    # entropy=entropy+(prob*math.log((prob_inv),2))
	    # print((symb2freq_r[p[0]]/(r.shape[0]*r.shape[1])))
	    # print(len(p[1]))
	    # print(entropy)
	    # print(prob_inv)
	    # print(math.log((prob_inv),2))

	    # np_r[p[0]]=symb2freq_r[p[0]]

	# print("Expected Length: "+str(exp_len_r))
	# print("Entropy: "+str(entropy))


	for x in g:
	    for y in x:
	        symb2freq_g[y] += 1

	huff_g = HuffmanEncoder(symb2freq_g)
	# print("Symbol\tWeight\tHuffman Code")
	for p in huff_g:
	    # print ("%s\t%s\t%s" % (p[0], symb2freq_g[p[0]], p[1]))
	    dic_g[p[0]]=p[1]
	    dic_g_rev[p[1]]=p[0]

	    # np_g[p[0]]=symb2freq_g[p[0]]


	for x in b:
	    for y in x:
	        symb2freq_b[y] += 1

	huff_b = HuffmanEncoder(symb2freq_b)
	# print("Symbol\tWeight\tHuffman Code")
	for p in huff_b:
	    # print ("%s\t%s\t%s" % (p[0], symb2freq_b[p[0]], p[1]))
	    dic_b[p[0]]=p[1]
	    dic_b_rev[p[1]]=p[0]

	    # np_b[p[0]]=symb2freq_b[p[0]]

	# print(len(dic_b))

	file_r=open("./output/r.txt","w+")
	file_g=open("./output/g.txt","w+")
	file_b=open("./output/b.txt","w+")
	total=0

	for x in r:
		for y in x:
			total=total+len(dic_r[y])
			# print(dic_r[y])
			file_r.write(dic_r[y])
			file_r.write('\n')

			# print(total)

	for x in g:
	    for y in x:
	        # file_r.write(str(y)+" ")
	        total=total+len(dic_g[y])
	        file_g.write(dic_g[y])
	        file_g.write('\n')

	for x in b:
	    for y in x:
	        # file_r.write(str(y)+" ")
	        total=total+len(dic_b[y])
	        file_b.write(dic_b[y])
	        file_b.write('\n')

	file_r.close()
	file_b.close()
	file_g.close()

	file_r=open("./output/r.txt","r")
	file_g=open("./output/g.txt","r")
	file_b=open("./output/b.txt","r")

	red=np.zeros((r.shape[0],r.shape[1]),dtype=np.uint8)
	blue=np.zeros((b.shape[0],b.shape[1]),dtype=np.uint8)
	green=np.zeros((g.shape[0],g.shape[1]),dtype=np.uint8)

	i=0
	j=0

	for line in file_r:
	    line=line.rstrip()
	    # line=line.split()[1]
	    red[i][j]=np.uint8(dic_r_rev[line])
	    # print(str(i)+" "+str(j)+" "+line+" "+str(dic_r_rev[line]))
	    if(j==r.shape[1]-1):
	        i=i+1
	        j=0
	    else:
	        j=j+1

	i=0
	j=0

	for line in file_g:
	    line=line.rstrip()
	    # line=line.split()[1]
	    green[i][j]=np.uint8(dic_g_rev[line])
	    # print(str(i)+" "+str(j)+" "+line+" "+str(dic_r_rev[line]))
	    if(j==g.shape[1]-1):
	        i=i+1
	        j=0
	    else:
	        j=j+1

	i=0
	j=0

	for line in file_b:
	    line=line.rstrip()
	    # line=line.split()[1]
	    blue[i][j]=np.uint8(dic_b_rev[line])
	    # print(str(i)+" "+str(j)+" "+line+" "+str(dic_r_rev[line]))
	    if(j==b.shape[1]-1):
	        i=i+1
	        j=0
	    else:
	        j=j+1

	combined=np.dstack((red,green,blue))
	
	print("PSNR: "+str(psnr(combined,file)))
	print("MSE: "+str(mse(combined,file)))
	print("RMSE: "+str(rmse(combined,file)))
	ratio=(original_file_size*8)/total
	print("Compression Ratio: "+str(ratio))

	combined=cv2.cvtColor(combined,cv2.COLOR_RGB2BGR)
	cv2.imwrite('./output/combined.png',combined)

	combined_d=np.dstack((r-red,g-green,b-blue))
	# print(np.min(r-red))
	# print(np.max(r-red))
	# print(np.min(g-green)
	# print(b-blue)

	combined_d=cv2.cvtColor(combined_d,cv2.COLOR_RGB2BGR)
	cv2.imwrite('./output/dif.png',combined_d)

	# xax=np.arange(0,256,1)

	# fig,(ax1,ax2,ax3)=plt.subplots(3,1)
	# # plt.plot(xax,np_r,'r-')
	# ax1.plot(xax,np_r,'r')
	# ax1.fill_between(xax, 0,np_r,facecolor='red')
	# ax1.grid(True)

	# ax2.plot(xax,np_g,'g')
	# ax2.fill_between(xax, 0,np_g,facecolor='green')
	# ax2.grid(True)

	# ax3.plot(xax,np_b,'b')
	# ax3.fill_between(xax, 0,np_b,facecolor='blue')

	# plt.xticks(np.arange(0,256, 25))
	# plt.grid(True)
	# plt.show()

	# print(original_file_size*8)
	# print(total)
	# arr.append(ratio)

def __init__():
	# file_list=['/home/singular/img/lena.png','/home/singular/img/peppers.png','/home/singular/img/1.jpg','/home/singular/img/3.jpg','/home/singular/img/4.jpg']

	# file_list_name=['Lena','Peppers','Mountain','Water','Street']
	# # file_list=['/home/singular/img/small4.png']
	# # file_list=['/home/singular/Desktop/1_1.jpeg']
	# # file_list_name=['Random']
	# counter=0

	# for file in file_list:
	main()
		# counter=counter+1


	# x=np.arange(1,6,1)
	# plt.grid(True,axis='y')

	# plt.bar(x, arr, align='center')
	# # plt.plot(x,ratios,'o-')
	# plt.xticks(x, file_list_name)
	# plt.title("Huffman Encoding(Ratio)")
	# plt.show()

	# print(arr)
__init__()
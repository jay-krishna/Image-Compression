import numpy as np 
import pandas as pd 
import cv2
from matplotlib import pyplot as plt

import os

def showImage(img):
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    plt.xticks([]),plt.yticks([])
    plt.show()

def showImage2(img,img1):
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(img1,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.show()

def selectQMatrix(qName):
    Q10 = np.array([[80,60,50,80,120,200,255,255],
                [55,60,70,95,130,255,255,255],
                [70,65,80,120,200,255,255,255],
                [70,85,110,145,255,255,255,255],
                [90,110,185,255,255,255,255,255],
                [120,175,255,255,255,255,255,255],
                [245,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255]])

    Q50 = np.array([[16,11,10,16,24,40,51,61],
                [12,12,14,19,26,58,60,55],
                [14,13,16,24,40,57,69,56],
                [14,17,22,29,51,87,80,62],
                [18,22,37,56,68,109,103,77],
                [24,35,55,64,81,104,113,92],
                [49,64,78,87,103,121,120,101],
                [72,92,95,98,112,100,130,99]])

    Q90 = np.array([[3,2,2,3,5,8,10,12],
                    [2,2,3,4,5,12,12,11],
                    [3,3,3,5,8,11,14,11],
                    [3,3,4,6,10,17,16,12],
                    [4,4,7,11,14,22,21,15],
                    [5,7,11,13,16,12,23,18],
                    [10,13,16,17,21,24,24,21],
                    [14,18,19,20,22,20,20,20]])
    if qName == "Q10":
        return Q10
    elif qName == "Q50":
        return Q50
    elif qName == "Q90":
        return Q90
    else:
        return np.ones((8,8)) #it suppose to return original image back


directory = '/home/singular/img/lena.png'
img = cv2.imread(directory)
# showImage(img)
height  = len(img) #one column of image
width = len(img[0]) # one row of image
# print (img.shape)
sliced_r = [] # new list for 8x8 sliced image
sliced_g = []
sliced_b = []
block = 8
currY = 0

r,g,b = cv2.split(img)

for i in range(block,height+1,block):
    currX = 0 #current X index
    for j in range(block,width+1,block):
        sliced_r.append(r[currY:i,currX:j]-np.ones((8,8))*128) #Extracting 128 from all pixels
        currX = j
    currY = i
    
# print("Size of the sliced image: "+str(len(sliced)))
# print("Each elemend of sliced list contains a "+ str(sliced[0].shape)+ " element.")
imf_r = [np.float32(r) for r in sliced_r]
DCToutput_r = []
for part in imf_r:
    currDCT = cv2.dct(part)
    DCToutput_r.append(currDCT)
DCToutput_r[0][0]

selectedQMatrix = selectQMatrix("Q90")
for ndct in DCToutput_r:
    for i in range(block):
        for j in range(block):
            ndct[i,j] = np.around(ndct[i,j]/selectedQMatrix[i,j])
DCToutput_r[0][0]

invList_r = []
for ipart in DCToutput_r:
    ipart
    curriDCT = cv2.idct(ipart)
    invList_r.append(curriDCT)
invList_r[0][0]
row = 0
rowNcol_r = []
for j in range(int(width/block),len(invList_r)+1,int(width/block)):
    rowNcol_r.append(np.hstack((invList_r[row:j])))
    row = j
res_r = np.vstack((rowNcol_r))

# showImage(res_r)
################################################################################################
currY = 0
for i in range(block,height+1,block):
    currX = 0 #current X index
    for j in range(block,width+1,block):
        sliced_g.append(g[currY:i,currX:j]-np.ones((8,8))*128) #Extracting 128 from all pixels
        currX = j
    currY = i
    
# print("Size of the sliced image: "+str(len(sliced)))
# print("Each elemend of sliced list contains a "+ str(sliced[0].shape)+ " element.")
imf_g = [np.float32(r) for r in sliced_g]
DCToutput_g = []
for part in imf_g:
    currDCT = cv2.dct(part)
    DCToutput_g.append(currDCT)
DCToutput_g[0][0]

selectedQMatrix = selectQMatrix("Q90")
for ndct in DCToutput_g:
    for i in range(block):
        for j in range(block):
            ndct[i,j] = np.around(ndct[i,j]/selectedQMatrix[i,j])
DCToutput_g[0][0]

invList_g = []
for ipart in DCToutput_g:
    ipart
    curriDCT = cv2.idct(ipart)
    invList_g.append(curriDCT)
invList_g[0][0]
row = 0
rowNcol_g = []
for j in range(int(width/block),len(invList_g)+1,int(width/block)):
    rowNcol_g.append(np.hstack((invList_g[row:j])))
    row = j
res_g = np.vstack((rowNcol_g))
# showImage(res_g)
#####################################################################################################
currY = 0
for i in range(block,height+1,block):
    currX = 0 #current X index
    for j in range(block,width+1,block):
        sliced_b.append(b[currY:i,currX:j]-np.ones((8,8))*128) #Extracting 128 from all pixels
        currX = j
    currY = i
    
# print("Size of the sliced image: "+str(len(sliced)))
# print("Each elemend of sliced list contains a "+ str(sliced[0].shape)+ " element.")
imf_b = [np.float32(r) for r in sliced_b]
DCToutput_b = []
for part in imf_b:
    currDCT = cv2.dct(part)
    DCToutput_b.append(currDCT)
DCToutput_b[0][0]

selectedQMatrix = selectQMatrix("Q90")
for ndct in DCToutput_b:
    for i in range(block):
        for j in range(block):
            ndct[i,j] = np.around(ndct[i,j]/selectedQMatrix[i,j])
DCToutput_b[0][0]

invList_b = []
for ipart in DCToutput_b:
    ipart
    curriDCT = cv2.idct(ipart)
    invList_b.append(curriDCT)
invList_b[0][0]
row = 0
rowNcol_b = []
for j in range(int(width/block),len(invList_b)+1,int(width/block)):
    rowNcol_b.append(np.hstack((invList_b[row:j])))
    row = j
res_b = np.vstack((rowNcol_b))
# showImage(res_b)

# print(r)
# print(res_r+128)

# print(g)
# print(res_g+128)

# print(b)
# print(res_b+128)
# res_r+=128
# res_g+=128
# res_b+=128

# res_r=res_r.astype(np.uint8)
# res_g=res_g.astype(np.uint8)
# res_b=res_b.astype(np.uint8)

# d_r=r-res_r
# d_g=g-res_g
# d_b=b-res_b

combined = np.dstack((res_r, res_g,res_b))
# diff=np.dstack((d_r, d_g,d_b))
# print(r)
# print(res_r)

# print(g)
# print(res_g)

# print(b)
# print(res_b)

# converted_image=cv2.cvtColor(combined,cv2.COLOR_BGR2RGB)
showImage2(r,res_r)
showImage2(g,res_g)
showImage2(b,res_b)
showImage(combined)
# cv2.imwrite("decoded.png",combined)
# cv2.imwrite("diff.png",diff)

# showImage(combined)
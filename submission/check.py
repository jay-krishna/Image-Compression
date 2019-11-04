import numpy as np
import cv2

t=cv2.imread('/home/singular/img/small16.png')
t=cv2.cvtColor(t, cv2.COLOR_BGR2YCR_CB)
a,b,c=cv2.split(t)

c_=np.zeros(a.shape,dtype=np.float)
c__=np.zeros(a.shape,dtype=np.uint)

lc=np.array([[17,18,24,47,99,99,99,99],
[18,21,26,66,99,99,99,99],
[24,26,56,99,99,99,99,99],
[47,66,99,99,99,99,99,99],
[99,99,99,99,99,99,99,99],
[99,99,99,99,99,99,99,99],
[99,99,99,99,99,99,99,99],
[99,99,99,99,99,99,99,99]])

for i in range(0,a.shape[0],8):
	for j in range(0,a.shape[1],8):
		x=b[i:i+8,j:j+8]
		c_[i:i+8,j:j+8]=(cv2.dct(x/255.0)*255)/lc
		c__[i:i+8,j:j+8]=cv2.idct((c_[i:i+8,j:j+8]*lc)/255.0)*255

print(c__)

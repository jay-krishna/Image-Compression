import numpy as np
import cv2
import matplotlib.pyplot as plt

file=cv2.imread('/home/singular/img/lena.png')

b,g,r=cv2.split(file)

converted_image=cv2.cvtColor(file,cv2.COLOR_BGR2YCR_CB)
y,cr,cb=cv2.split(converted_image)

# y_=np.zeros((y.shape),dtype=np.uint8)
# cr_=np.zeros((y.shape),dtype=np.uint8)
# cb_=np.zeros((y.shape),dtype=np.uint8)

# print(type(y_[0][0]))
# print(y_)

y_=np.round(y/1.5)
cr_=np.round(cr-5)
cb_=np.round(cb-5)

# y_=10*y_
# cr_=cr_*5
# cb_=cb_*5

# print(y_)

y_=y_.astype('uint8')
# cr_=cr_.astype('uint8')
# cb_=cb_.astype('uint8')

# print(type(y_[0][0]))

combine1=np.dstack((y_,cr,cb))
combine2=np.dstack((y,cr_,cb_))

# print(type(y_[0][0]))

combine1=cv2.cvtColor(combine1,cv2.COLOR_YCR_CB2BGR)
combine2=cv2.cvtColor(combine2,cv2.COLOR_YCR_CB2BGR)

cv2.imwrite('./y.png',y)
cv2.imwrite('./cb.png',cb)
cv2.imwrite('./cr.png',cr)
cv2.imwrite('./color.png',combine2)
cv2.imwrite('./light.png',combine1)
cv2.imwrite('./r.png',r)
cv2.imwrite('./g.png',g)
cv2.imwrite('./b.png',b)


# cv2.imwrite('./c.png',combine2)

# print(y)
# print(y_)
# print("$$$$$$$$$$$$$$$$")
# print(cr)
# print(cr_)
# print(cr.shape)
# print(cr_.shape)
# print("$$$$$$$$$$$$$$$$$")
# print(cb)
# print(cb_)

# plt.plot(y)
# plt.show()
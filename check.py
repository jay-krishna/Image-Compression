import numpy as np

l=[[1,2,2,3],[1,1,2,3],[1,2,3,3],[3,3,3,3],[4,1,4,1]]
ll=np.array(l)
# count=0
# f=[]

# for x in range(0,ll.shape[0]):
# 	temp=[]
# 	count=0
# 	for y in range(0,ll.shape[1]-1):
# 		_=ll[x][y]
# 		__=ll[x][y+1]

# 		# print(str(_)+" "+str(__))

# 		if _==__:
# 			count=count+1
# 		else:
# 			if count>0:
# 				temp.append((_,count+1))
# 				# temp.append(__)
# 				count=0
# 			else:
# 				count=0
# 				temp.append(_)

# 	if(count>0):
# 		temp.append((_,count+1))
# 	else:
# 	# 	temp.append(_)
# 		temp.append(__)
# 	f.append(temp)

# print(ll)
# print(f)
# recon=[]
# for x in f:
# 	ttt=[]
# 	for y in x:
# 		if(isinstance(y,tuple)):
# 			for i in range(0,y[1]):
# 				ttt.append(y[0])
# 		else:
# 			ttt.append(y)

# 	recon.append(ttt)

# print(recon)
# nrecon=np.array(recon)
# print(nrecon.shape)

class HeapNode:
	def __init__(self, char, freq):
		self.char = char
		self.freq = freq
		self.left = None
		self.right = None

	def __lt__(self, other):
		if(other == None):
			return -1
		if(not isinstance(other, HeapNode)):
			return -1
		return self.freq > other.freq

class HuffmanCode:
	"""docstring for HuffmanCode"""
	def __init__(self, arg):
		self.heap = []
		self.codes = {}
		


print(ll)
value=np.zeros(10)
print(value)

for x in ll:
	for y in x:
		value[y]=value[y]+1

print(value)
value=value/(ll.shape[0]*ll.shape[1])
print(value)
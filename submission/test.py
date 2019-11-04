import numpy as np
from zigzagf import zigzag,inverse_zigzag
from heapq import heappush, heappop, heapify
from collections import defaultdict,Counter

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

# ll=np.array([[3,2,2,3,5,8,10,12],
# 	        [2,2,3,4,5,12,12,11],
# 	        [3,3,3,5,0,0,0,0],
# 	        [3,3,4,6,0,0,0,0],
# 	        [0,0,0,0,0,0,0,0],
# 	        [0,0,0,0,0,0,0,0],
# 	        [0,0,0,0,0,0,0,0],
# 	        [0,0,0,0,0,0,0,0]])

ll=np.array([[68,4,0,-4,-4,0,-1,-1],
 [-1,-5,1,2,0,-2,0,1],
 [-1,-3,1,-2,1,-1,-1,1],
 [0,3,0,0,2,-1,1,0],
 [1,0,0,0,0,0,0,0],
 [-2,0,-1,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [-1,0,0,0,0,0,0,0]])

# print(ll)
print(ll)
r=zigzag(ll)
print(r)
count_r=0
final_r=[]

for i in range(r.shape[0]-1):
	r_=r[i]
	r__=r[i+1]

	# print(str(r_)+" "+str(r__))

	if r_==r__:
		count_r=count_r+1
	else:
		if count_r>0:
			final_r.append((r_,count_r+1))
			# temp_r.append(_)
			count_r=0
			# count=count+3
		else:
			count_r=0
			final_r.append(r_)
			# count=count+1

if count_r>0:
	final_r.append((r_,count_r+1))
else:
	final_r.append(r__)

# print(final_r)
# print(zigzag(ll).shape)
# print(inverse_zigzag(zigzag(ll),ll.shape[0],ll.shape[1]))

symb2freq_r = defaultdict(int)
# huff_r = HuffmanEncoder(symb2freq_r)

for _ in final_r:
	if(isinstance(_, tuple)):
		symb2freq_r[_[0]]+=_[1]
		symb2freq_r[_[1]]+=1
	else:
		symb2freq_r[_]+=1

# print(symb2freq_r)
huff_r = HuffmanEncoder(symb2freq_r)
dic_r={}
print("Symbol\tFrequency\tHuffman Code\tProbability")
for p in huff_r:
	dic_r[p[0]]=p[1]
	# print(p)
	print ("%s\t%s\t\t%s\t\t%s" % (int(p[0]),symb2freq_r[p[0]],p[1],symb2freq_r[p[0]]/(ll.shape[0]*ll.shape[1])))
    # print("%s\t%s\t%s"%(p[0],symb2freq_r[p[0]],symb2freq_r[p[0]]/(ll.shape[0]*ll.shape[1])))

total=0
# for x in ll:
# 	for y in x:
# 		total+=len(dic_r[y])

# print(total)

for _ in final_r:
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
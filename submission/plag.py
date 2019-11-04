from heapq import heappush, heappop, heapify
from collections import defaultdict,Counter
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import math
 
def encode(symb2freq):
    heap=[]

    for _,__ in symb2freq.items():
        heap.append([__,[_,""]])
    heapify(heap)

    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)

        print(lo)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
 
l=[[1,2,2,3],[1,1,2,3],[1,2,3,3],[3,3,3,3],[4,1,4,1]]
symb2freq = defaultdict(int)
for x in l:
    for y in x:
        symb2freq[y] += 1
# in Python 3.1+:
# symb2freq = Counter(l)
huff = encode(symb2freq)
print("Symbol\tWeight\tHuffman Code")
dic={}
for p in huff:
    print ("%s\t%s\t%s" % (p[0], symb2freq[p[0]], p[1]))
    dic[p[0]]=p[1]

print(dic)
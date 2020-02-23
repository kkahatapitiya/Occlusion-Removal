import os
import numpy as np
import cv2
import sys
#sys.path.insert(0, '/home/kumarak/Desktop/campus_temp/pred2/')
#import get_dataset_colormap

read="./all_at_100_nocol/"
gtread=open("./thinglabels.txt").readlines()
gt={}
#print(gtread)
for i in gtread:
	gt[int(i.split(':')[0])]=i.split(':')[1][1:-1]
#print(gt)

#map=get_dataset_colormap.create_label_colormap()
#list=[(map[i],i) for i in range(0,len(map))]
list=[]
for filename in os.listdir(read):
	#print(filename)
	if filename.endswith('.png'):
		img=cv2.imread(read+filename)
		classes=[gt[i] for i in np.unique(img) if i!=255]
		list.append((filename,classes))

for i in sorted(list):
	print(i)

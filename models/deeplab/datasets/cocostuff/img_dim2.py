import os
from PIL import Image

direct='/Users/kumarakahatapitiya/Desktop/Datasets/VOCdevkit/VOC2012/JPEGImages/'

max_w=0;max_h=0;
for filename in os.listdir(direct):
	#print(filename) 
	if filename.endswith(".jpg"):
		#print(filename) 
		im = Image.open(direct+filename)
		w, h = im.size
		if w>max_w:
			max_w=w
		if h>max_h:
			max_h=h

print(max_w,max_h)
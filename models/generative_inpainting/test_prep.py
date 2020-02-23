import os
import cv2
import numpy as np
import sys
sys.path.insert(0, '/Users/kumarakahatapitiya/Desktop/work temp/scripts/')
import get_dataset_colormap
from random import randint

gt_things_read=open("/Users/kumarakahatapitiya/Desktop/work temp/thinglabels_updated.txt").read().splitlines()
map=get_dataset_colormap.create_label_colormap()
list=[(map[i][::-1],i) for i in range(0,len(map))]
gt={}			# thing class number ---> name
for i in gt_things_read:
	gt[int(i.split(':')[0])]=i.split(':')[1][1:]

tempdir='/Users/kumarakahatapitiya/Downloads/generative_inpainting-master/my_test/val_temp'
imagedir='/Users/kumarakahatapitiya/Downloads/generative_inpainting-master/my_test/val_image'
imagetempdir='/Users/kumarakahatapitiya/Downloads/generative_inpainting-master/my_test/val_image_temp'
maskdir='/Users/kumarakahatapitiya/Downloads/generative_inpainting-master/my_test/val_mask'
'''
a={'000011_image.png':1,'000009_image.png':1,'000006_image.png':5,
	'000005_image.png':62,'000013_image.png':72}
'''
for filename in os.listdir(imagetempdir):
	if filename.endswith('.png'):
		things=[]
		img=cv2.imread(os.path.join(tempdir,filename))
		image=cv2.imread(os.path.join(imagetempdir,filename))
		for i in range(0,len(list)):
			ind=np.where((img == list[i][0]).all(axis = 2))
			if len(ind[0])!=0:
				things.append({'class':i,'indices':ind})
		out=np.zeros(np.shape(img))
		ind=randint(0,len(things)-1)
		ind=max(1,ind)
		if ind<len(things):
			print(filename,things[ind]['class'])		
			out[things[ind]['indices']]=255

		cv2.imwrite(os.path.join(maskdir,filename),cv2.resize(out,(256,256)))
		cv2.imwrite(os.path.join(imagedir,filename),cv2.resize(image,(256,256)))
'''
x='/Users/kumarakahatapitiya/Downloads/generative_inpainting-master/my_test/val_temp/'
for filename in os.listdir(x):
	if filename.endswith('.png'):
		img=cv2.imread(os.path.join(x,filename))
		cv2.imwrite(os.path.join(x,filename[:-15]+filename[-4:]),img)
'''
import os
import numpy as np
import cv2
import sys
sys.path.insert(0, './scripts/')
import get_dataset_colormap

read="./test_col/"
save="./test_nocol/"

map=get_dataset_colormap.create_pascal_label_colormap()
#list=[(map[i],i) for i in range(0,len(map))]
list=[(map[i][::-1],i) for i in range(0,len(map))]

for filename in os.listdir(read):
	print(filename)
	if filename.endswith('.png'):
		img=cv2.imread(read+filename)
		img2=np.zeros(np.shape(img))
		img2.fill(255)
		#print(img)
		for i in range(0,len(list)):
			#img[img==list[i][0]]=i
			#print(i,np.where((img == list[i][0]).all(axis = 2)))
			img2[np.where((img == list[i][0]).all(axis = 2))] = [i,i,i]
			cv2.imwrite(save+filename[:-4]+'_nocol'+filename[-4:],img2)
			#print(img[np.where((img == list[i][0]))])
	
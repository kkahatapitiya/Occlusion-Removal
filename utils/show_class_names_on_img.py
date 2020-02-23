import os
import numpy as np
import cv2
import sys
sys.path.insert(0, './scripts/')
import get_dataset_colormap
from random import randint

read="./300_pred_all/"
save="./300_pred_all/"
gtread=open("./thinglabels_updated.txt").readlines()

gt={}
print(gtread)
for i in gtread:
	#print(i.split(':'))
	gt[int(i.split(':')[0])]=i.split(':')[1][1:-1]
#print(gt)

map=get_dataset_colormap.create_label_colormap()
list=[(map[i][::-1],i) for i in range(0,len(map))]

font                   = cv2.FONT_HERSHEY_COMPLEX
#location               = (10,500)
fontScale              = 0.7
fontColor              = (255,255,255)
lineType               = 2

for filename in os.listdir(read):
	print(filename)
	if filename.endswith('.png'):
		img=cv2.imread(read+filename)
		#img2=np.zeros(np.shape(img))
		#img2.fill(255)
		#print(img)
		writearr=[]
		for i in range(0,len(list)):
		#for i in range(0,91):
			ind=np.where((img == list[i][0]).all(axis = 2))
			if len(ind[0])!=0:
				#k=randint(0,len(ind))
				writearr.append({'class':gt[i], 'x':int(ind[1][len(ind[1])/2+len(ind[1])/10]), 'y':int(ind[0][len(ind[1])/2+len(ind[1])/10])})
			#img[img==list[i][0]]=i
			#img2[np.where((img == list[i][0]).all(axis = 2))] = [i,i,i]
			#cv2.imwrite(save+filename,img2)
			#print(img[np.where((img == list[i][0]))])
		#classes=[gt[i] for i in np.unique(img2) if i!=255]
		for i in range(0,len(writearr)):
			cv2.putText(img,writearr[i]['class'], (writearr[i]['x'],writearr[i]['y']), font, fontScale, fontColor, lineType)
		cv2.imwrite(save+filename,img)

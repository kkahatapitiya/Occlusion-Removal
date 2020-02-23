import os
import numpy as np
import cv2
import cPickle as pickle

testlist=open('./train.txt','r+').read().splitlines()

imdir='./dataset/MScoco_stuff/images/images'
shape=256
dic={}

num_samples=len(testlist)
sample_per_itr=10000
itr=num_samples/sample_per_itr

for p in range(0,itr):
	file=open('./train_'+str(p),'wb')
	imgdata=np.empty((sample_per_itr, 3, shape,shape), dtype='uint8')
	classdata=np.empty((sample_per_itr, 91), dtype='uint8')

	for i in range(0,sample_per_itr):
		#print(type(i),i)
		imname,stuff=testlist[i].split('\t')
		print(p,i,imname,stuff)
		stuff=[int(k) for k in stuff.split(',')]
		#print(stuff)
		gt=np.zeros(91, dtype='uint8')
		for j in stuff:
			gt[j]=1
		img=(cv2.resize(cv2.imread(os.path.join(imdir,imname+'.jpg')),(shape,shape))).transpose(2,0,1)
		#print(np.shape(img),img)
		#print(np.shape(imgdata[0]),np.shape(img))
		imgdata[i]=img
		classdata[i]=gt

	#print(imgdata)
	#print(classdata)
	dic['image']=imgdata
	dic['class']=classdata
	#file.write(dic)
	pickle.dump(dic,file,pickle.HIGHEST_PROTOCOL)
	#dic.tofile(file)
	#classdata.tofile(file)


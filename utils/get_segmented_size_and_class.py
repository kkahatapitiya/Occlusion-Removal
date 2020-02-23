import numpy as np
import cv2
import sys
sys.path.insert(0, './scripts/')
import get_dataset_colormap

image_map_list=open('./vis_full_150kepochs/image_map.txt').read().splitlines()
gt_things_read=open("./thinglabels_updated.txt").read().splitlines()
deeplab_dir='./vis_full_150kepochs/'
writefile='./vis_full_150kepochs/segment_size_and_class.txt'
#entries=[i.split('\t')[0] for i in readfile]

image_map={}	# coco image name ---> deeplab image name 4895
for i in image_map_list:
	image_map[i.split('\t')[1]]=i.split('\t')[0]

map=get_dataset_colormap.create_label_colormap()
list=[(map[i][::-1],i) for i in range(0,len(map))]
gt={}			# thing class number ---> name
for i in gt_things_read:
	gt[int(i.split(':')[0])]=i.split(':')[1][1:]

for name in image_map.keys():
	#if name not in entries:
	things=''
	things_image=deeplab_dir+'predictions/'+image_map[name][:-5]+'prediction.png'
	img=cv2.imread(things_image)
	im_size=np.shape(img)[0]*np.shape(img)[1]
	for i in range(0,len(list)):
		ind=np.where((img == list[i][0]).all(axis = 2))
		if len(ind[0])!=0:
			things+=gt[i]+','
			things+=str(np.shape(ind)[1])+','
	#print(things)
	if things!='':
		things=things[:-1].split(',',2)
		if len(things)>2:
			things=things[2]
		else:
			things=''
	f=open(writefile,'a')
	f.write(name+','+str(im_size)+','+things+'\n')
	f.close()
	print(name,things)



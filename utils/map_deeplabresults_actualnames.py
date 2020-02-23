import cv2
import numpy as np

deeplab_dir='./vis_full_150kepochs/images/'
actual_img_dir='./dataset/MScoco_stuff/images/images/'
images=[i.split('\t')[0] for i in open('./val.txt').read().splitlines()]
writefile='./vis_full_150kepochs/image_map.txt'
img_dic={}
for i in images:
	img_dic[i]=cv2.imread(actual_img_dir+i+'.jpg')

for filename in os.listdir(deeplab_dir):
	img=cv2.imread(os.path.join(deeplab_dir,filename))
	print(filename)
	for i in range(0,len(img_dic.values())):
		#if(np.array_equal(img_dic.values()[i],img)):
		shape=np.shape(img)
		if(shape==np.shape(img_dic.values()[i])):
			#print(i,shape,img_dic.keys()[i])
			confidence=np.count_nonzero((((img-img_dic.values()[i])+10)<20)*1)/float(shape[0]*shape[1]*shape[2])
			if(confidence>0.75):
				f=open(writefile,'a')
				f.write(filename[:-4]+'\t'+img_dic.keys()[i]+'\n')
				print(i,img_dic.keys()[i],confidence,'**********')
				f.close()
				break
#np.count_nonzero((((img-img_dic.values()[i])+10)<20)*1)/float(np.shape(img)[0]*np.shape(img)[1]*np.shape(img)[2])
#im=deeplab_dir+'images/004951_image.png'

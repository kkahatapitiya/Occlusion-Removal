import json
import numpy as np
import cv2

embeddings='./coco_embeddings.json'
classmap='./class_map.json'
file=open('./b_cosine_w2v_v3_sw3.txt','w')

with open(embeddings) as f:
    class_embed = json.load(f)

with open(classmap) as f:
    classmap = json.load(f)

maxd=0
#maxval=1.483
#minval=0.613

#new
dist_max=1.687
dist_min=0.667

cosine_max=0.777
cosine_min=-0.423


for i in range(0,len(class_embed.keys())):
	for j in range(i+1,len(class_embed.keys())):
		one=np.array(class_embed[class_embed.keys()[i]])
		two=np.array(class_embed[class_embed.keys()[j]])

		#print(np.linalg.norm(one,ord=2),np.linalg.norm(one,ord=2))
		dist1=np.dot(one,two)
		dist2=np.linalg.norm((one-two),ord=2)

		dist1_norm=float("{0:.3f}".format((dist1-cosine_min)/(cosine_max-cosine_min)))
		dist2_norm=float("{0:.3f}".format((dist2-dist_min)/(dist_max-dist_min)))

		#if dist>maxd:
		#	maxd=dist
		print(class_embed.keys()[i],class_embed.keys()[j],dist1_norm,dist2_norm)
		file.write(str(dist1_norm)+'\t'+str(1-dist2_norm)+'\t'+class_embed.keys()[i]+', '+class_embed.keys()[j]+'\n')
print(maxd)

'''
num_cls=len(classmap.keys())
size=10
keys=[str(i) for i in range(1,num_cls+1)]
img=np.zeros((size*num_cls,size*num_cls,3),'int32')
for i in range(0,num_cls):
	for j in range(i+1,num_cls-1):
		print(keys[i],keys[j])
		try:
			one=np.array(class_embed[classmap[keys[i]]['w2vclass']])
		except KeyError:
			one=np.array(class_embed[classmap[keys[i]]['actualclass']])
		try:
			two=np.array(class_embed[classmap[keys[j]]['w2vclass']])
		except KeyError:
			two=np.array(class_embed[classmap[keys[j]]['actualclass']])

		print(i,j)
		dist=float("{0:.3f}".format(np.linalg.norm((one-two),ord=2)))
		dist=float("{0:.3f}".format((dist-minval)/(maxval-minval)))
		col=int(dist*255)
		img[j*size:(j+1)*size,i*size:(i+1)*size,:]=col
cv2.imwrite('/Users/kumarakahatapitiya/Desktop/imgc.jpg',img)
#cv2.waitKey(0)
'''



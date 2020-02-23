import json
import numpy as np
import cv2
import sys
sys.path.insert(0, './scripts/')
import get_dataset_colormap


embeddings='./models/word2vec-coco/coco_embeddings.json'
classmap='./models/word2vec-coco/class_map.json'				# class map between coco classes (182) and w2v classes (169)
deeplab_dir='./vis_full_150kepochs/'							# deeplab out images and predictions 4953

images=[i.split('\t')[0] for i in open('./textfiles/val.txt').read().splitlines()]					# images in val.txt 4942
class_names=[i.split(':')[1][1:] for i in open('./textfiles/stufflabels.txt').read().splitlines()]	# 0 banner - 90 wood
image_map_list=open('./vis_full_150kepochs/image_map.txt').read().splitlines()					# deeplab image name [tab] coco image name 4895
gt_things_read=open("./thinglabels_updated.txt").read().splitlines()								# for getting thing names from deeplab out 0 background - 91 hair brush
y_pred=np.loadtxt('./pred.txt')															# stuff predictions val (4864, 91) matrix
thing_predict_file=open('./vis_full_150kepochs/predicted_thingclasses.txt').read().splitlines()	# deeplab predicted thing classes
distances_file=open('./vis_full_150kepochs/distances_magupdated_new.txt','w')

map=get_dataset_colormap.create_label_colormap()
list=[(map[i][::-1],i) for i in range(0,len(map))]

gt={}			# thing class number ---> name
for i in gt_things_read:
	gt[int(i.split(':')[0])]=i.split(':')[1][1:]

image_map={}	# coco image name ---> deeplab image name 4895
for i in image_map_list:
	image_map[i.split('\t')[1]]=i.split('\t')[0]

thing_prediction={}
for i in thing_predict_file:
	enrty=i.split('\t')
	if len(enrty)>1:
		thing_prediction[i.split('\t')[0]]=i.split('\t')[1]
	else:
		thing_prediction[i.split('\t')[0]]=''

hard_thres=0.1
w2v_thres=0.5

with open(embeddings) as f:
    class_embed = json.load(f)

with open(classmap) as f:
    classmap = json.load(f)

y_trunc=(y_pred>hard_thres)*1.0 # 4864 vectors

#get predicted thing and stuff classes for each image
predictions={}
p=0
for name in image_map.keys():
	if images.index(name)>4723: 	# avoid index overflow 4864
		continue
	stuff_vector=y_trunc[images.index(name)]
	pred_stuff_names=[class_names[k] for k in np.nonzero(stuff_vector)[0]]
	pred_thing_names=thing_prediction[name].split(',')
	p+=1
	#print(p,name,pred_thing_names,pred_stuff_names)
	predictions[name]=[pred_thing_names,pred_stuff_names]

#calculate unrelated thing classes
all_dist=[]
maxdist=0
maxvalue_prev=1.483
minvalue_prev=0.613

maxvalue_new=1.687
minvalue_new=0.667

for i in predictions.keys():
	if(predictions[i][0]!=[''] and predictions[i][1]!=[]):
		#print(predictions[i])
		things=[classmap[j]['w2vclass'] for j in predictions[i][0]] #classmap[j]['classnum']
		stuff=[classmap[j]['w2vclass'] for j in predictions[i][1]]
		stuff_embed=[]
		string=i+' : '
		for j in stuff:
			string+=j+','
			stuff_embed.append(class_embed[j])
		stuff_embed=np.array(stuff_embed,dtype=np.float32)
		centroid=np.mean(stuff_embed,axis=0)
		#print(np.mean(centroid,axis=0))
		#print(np.shape(centroid))
		distances={}
		string='{0: <50}'.format(string[:-1])+': '
		for k in things:
			#dist=float("{0:.3f}".format(np.dot(class_embed[k],centroid)))
			dist=float("{0:.3f}".format((np.linalg.norm((class_embed[k]-centroid),ord=2)-minvalue_new)/(maxvalue_new-minvalue_new)))
			if dist>maxdist:
				maxdist=dist
			#string+=k+'-'+str(dist)+' , '
			distances[k]=dist
			all_dist.append(dist)
		for l in (sorted(distances.values())):
			string+=str(l)+'-'+distances.keys()[distances.values().index(l)]+' , '
		string=string[:-2]+'\n'
		print(string)
		distances_file.write(string+'\n')
distances_file.close()

print(maxdist)

'''
writefile='/Users/kumarakahatapitiya/Downloads/word2vec-coco-master/class_map_alt.json'
actualclassfile=open('/Users/kumarakahatapitiya/Desktop/codebase/textfiles/things_stuff_labels.txt').read().splitlines()
w2vclassfile=open('/Users/kumarakahatapitiya/Downloads/words.txt').read().splitlines()
dic={}
for i in actualclassfile:
	#dic[int(i.split(':')[0])]={'actualclass':i.split(':')[1][1:]}
	dic[i.split(':')[1][1:]]={'classnum':int(i.split(':')[0])}
#print(dic)

for j in w2vclassfile:
	w2v=j.split(',')[0]
	if len(j.split(','))>1:
		act=j.split(',')[1].split('/')
		for k in act:
			#print(dic.keys()[dic.values().index({'actualclass':k})])
			#dic[dic.keys()[dic.values().index({'actualclass':k})]]['w2vclass']=w2v
			dic[k]['w2vclass']=w2v
	else:
		#dic[dic.keys()[dic.values().index({'actualclass':j})]]['w2vclass']=j
		dic[w2v]['w2vclass']=w2v
print(dic)
for i in dic:
	print dic[i]
with open(writefile, 'w') as outfile:
    json.dump(dic, outfile)
'''

'''
	things_image=deeplab_dir+'predictions/'+image_map[name][:-5]+'prediction.png'
	img=cv2.imread(things_image)
	for i in range(0,len(list)):
		ind=np.where((img == list[i][0]).all(axis = 2))
		if len(ind[0])!=0:
			things.append(gt[i])
'''




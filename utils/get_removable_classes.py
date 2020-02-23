import os
from shutil import copyfile

thres=0.25
thres_new=0.7
cnt=0
im={}
inputdir='./dataset/MScoco_stuff/images/val2017'
outputdir='./new_test/in'
dist=open('./vis_full_150kepochs/distances_magupdated_new.txt').read().splitlines()
for line in dist:
	if line!='':
		line=line.split(':')
		name=line[0][:-1]
		val=line[2].replace('-',',').split(',')
		#print(val)
		thing=[val[i].replace(' ', '') for i in range(1,len(val),2)]
		val=[float(val[i]) for i in range(0,len(val),2)]		
		#print(val)
		#if sorted(val)[-1]>thres_new:
		#	print(name)
		k=0;temp=[]
		#while k<len(val) and val[k]<thres:
		while k<len(val):
			if val[k]>thres_new:
				temp.append(thing[k])
				#print(temp)
			k+=1
		if len(temp)!=0:
			#print(name,temp)
			im[name]=temp
			cnt+=1
#print(im)
#print(len(im.keys()))

size=open('./vis_full_150kepochs/segment_size_and_class.txt').read().splitlines()
dic={}
for i in size:
	line=i.split(',')
	dic[line[0]]={'res':line[1],'class':line[2:]}
#print(dic)

num=0
feasible=[]
for i in im.keys():
	areas=[k.replace(' ', '') for k in dic[i]['class']]
	things=im[i]
	#print(things,areas)
	removables=[]
	for j in things:
		ind=areas.index(j)
		precent=float(areas[ind+1])/float(dic[i]['res'])
		if precent>0.05 and precent<0.15:
			#print(i,j)
			removables.append(j)
	if len(removables)!=0:
		feasible.append((i,removables))
		num+=1
print(num)
'''
for i in range(0,len(feasible)):
	print(feasible[i])
	copyfile(os.path.join(inputdir,feasible[i][0]+'.jpg'),os.path.join(outputdir,feasible[i][0]+'.jpg'))
'''
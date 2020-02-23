import os
import numpy as np
#import cv2

supercls=['water','ground','solid','sky','plant','structural','building',
			'food','textile','furniture','window','floor','ceiling','wall','rawmaterial']

stufflist=open('./stufflabels.txt','r+').read().splitlines()

ownership=[8,8,4,6,6,4,9,5,14,11,
			12,12,8,8,3,9,9,8,9,1,
			9,5,11,11,11,11,11,4,0,7,
			7,9,4,1,1,2,6,4,9,8,
			14,9,4,2,1,8,5,14,1,8,
			4,14,1,1,5,1,0,1,2,6,
			8,7,1,0,9,3,6,1,2,9,
			2,4,5,9,6,8,8,4,7,13,
			13,13,13,13,13,13,0,0,10,10,2]

classes=[{'class':stufflist[i].split(':')[1][1:],'classnum':int(stufflist[i].split(':')[0]),'superclass':supercls[ownership[i]],'superclassnum':ownership[i]} for i in range(0,len(stufflist))]

#for i in sorted(classes, key=lambda k: k['superclass']) :
#	print(i['superclass']+'\t'+i['class'])


read=open('./val.txt','r')
write=open('./val2.txt','w')

sortedclass=sorted(classes, key=lambda k: k['classnum'])
for line in read:
        #print(line.split('\t'))
        l=line.split('\t')
        cl=[int(i) for i in l[1].split(',')]
        superclass=set([str(sortedclass[i]['superclassnum']) for i in cl])                           
        write.write(l[0]+'\t'+",".join(superclass)+'\n')
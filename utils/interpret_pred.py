import numpy as np
import collections

gt=[i.split('\t')[1] for i in open('./val.txt').read().splitlines()]
images=[i.split('\t')[0] for i in open('./val.txt').read().splitlines()]
y_pred=np.loadtxt('./pred.txt')
class_names=[i.split()[1] for i in open('./stufflabels.txt').read().splitlines()]

'''
for t in [0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15]:
        y_trunc=(y_pred>t)*1.0
        corr1=0;wro1=0;
        for i in range(0,np.shape(y_trunc)[0]):
                for j in np.nonzero(y_trunc[i])[0]:
                        if j in [int(k) for k in gt[i].split(',')]:
                                corr1+=1
                        else:
                                wro1+=1
        corr2=0;wro2=0;
        for i in range(0,np.shape(y_trunc)[0]):
                for j in [int(k) for k in gt[i].split(',')]:
                        if j in list(np.nonzero(y_trunc[i])[0]):
                                corr2+=1
                        else:
                                wro2+=1
        print('pred in gt',corr1,wro1,float(corr1)/(corr1+wro1),'\tgt in pred',corr2,wro2,float(corr2)/(corr2+wro2))
'''
t=0.1
y_trunc=(y_pred>t)*1.0

writefile=open('./predictions_interpret.txt','w')
dic=[]
for i in range(0,np.shape(y_trunc)[0]):
	pred_names=[class_names[k] for k in np.nonzero(y_trunc[i])[0]]
	gt_names=[class_names[int(k)] for k in gt[i].split(',')]
	#print(set(gt_names).intersection(set(pred_names)))
	str_pred=','.join(pred_names)
	str_gt=','.join(gt_names)
	dic.append({'im':images[i],'info':str_pred+'\n'+str_gt})
	#print(images[i]+'\n'+str_pred+'\n'+str_gt+'\n\n')
	#writefile.write(images[i]+'\n'+str_pred+'\n'+str_gt+'\n\n')

for i in sorted(dic, key=lambda k: k['im']):
	print(i['im']+'\n'+i['info']+'\n\n')
	writefile.write(i['im']+'\n'+i['info']+'\n\n')


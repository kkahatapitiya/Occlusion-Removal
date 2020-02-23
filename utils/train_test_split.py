

read=open('./stuff_gt.txt')
ref=open('./dataset/MScoco_stuff/annotations/stuffthingmaps_trainval2017/val.txt').read().splitlines()

trainlist=open('./train.txt','w+')
testlist=open('./val.txt','w+')

for line in read:
	#print(line.split('\t'))
	l=line.split('\t')
	if len(l)==2:
		if l[0] in ref:
			testlist.write(line)
		else:
			trainlist.write(line)
#print(ref)
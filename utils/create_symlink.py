import os

trainlist=open('./dataset/MScoco_stuff/annotations/stuffthingmaps_trainval2017/val.txt').read().splitlines()
readdir='./dataset/MScoco_stuff/images/images/'
writedir='./dataset/MScoco_stuff/images/val/'

for filename in os.listdir(readdir):
        if filename[:-4] in trainlist:
                os.symlink(readdir+filename,writedir+filename)
import os
from shutil import copyfile

read='/Users/kumarakahatapitiya/Desktop/Datasets/MScoco_stuff/images/images/'
save='/Users/kumarakahatapitiya/Desktop/Datasets/MScoco_stuff/images/val2017/'

#file='/Users/kumarakahatapitiya/Desktop/Datasets/MScoco_stuff/annotations/stuffthingmaps_trainval2017/train.txt'
file='/Users/kumarakahatapitiya/Desktop/Datasets/MScoco_stuff/annotations/stuffthingmaps_trainval2017/val.txt'


with open(file) as f:
    lines = f.read().splitlines()
#print(lines)
i=0
for filename in os.listdir(read):
    if filename.split('.')[0] in lines: 
        print(filename)
        i+=1
        copyfile(read+filename, save+filename)
print(i)
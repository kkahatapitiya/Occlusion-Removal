import os

#direct='/Users/kumarakahatapitiya/Desktop/Datasets/MScoco_stuff/annotations/stuffthingmaps_trainval2017/train2017/'
direct='/Users/kumarakahatapitiya/Desktop/Datasets/MScoco_stuff/annotations/stuffthingmaps_trainval2017/val2017/'

#file='/Users/kumarakahatapitiya/Desktop/Datasets/MScoco_stuff/annotations/stuffthingmaps_trainval2017/train.txt'
file='/Users/kumarakahatapitiya/Desktop/Datasets/MScoco_stuff/annotations/stuffthingmaps_trainval2017/val.txt'

f = open(file,'w+') 

for filename in os.listdir(direct):
    if filename.endswith(".png"): 
        f.write(filename.split('.')[0]+'\n')

f.close()
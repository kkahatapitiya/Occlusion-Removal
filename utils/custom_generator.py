import os
import cv2
import numpy as np
import random

train_list='./test_img/train.txt'
val_list='./test_img/val.txt'
img_dir='./dataset/MScoco_stuff/images/images/'

batch_size=128
shape=(512,512)

def generate_data(directory, shape, batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    #print('fn')
    dat_list=open(directory).read().splitlines()
    train={}; img=[]; i=0;
    for j in dat_list:
        train[j.split('\t')[0]]=j.split('\t')[1]
        img.append(j.split('\t')[0])

    while True:
        image_batch = []; class_batch = [];
        for b in range(batch_size):
            if i == len(img):
                i = 0; random.shuffle(img);
            img_sample = img[i]; class_sample = train[img[i]];
            i += 1
            #print(img_dir+img_sample)
            image = cv2.resize(cv2.imread(img_dir+img_sample+'.jpg'), shape) #.transpose(2,0,1)
            image_batch.append((image.astype(float) - 128) / 128)

            gt=np.zeros(91, dtype='float')
            for k in class_sample.split(','):
                gt[int(k)]=1.
            class_batch.append(gt)
        #print(np.shape(image_batch),np.shape(class_batch))
        yield np.array(image_batch),np.array(class_batch)


#print('main',shape,batch_size)
#mygenerator=generate_data(train_list,shape,batch_size)
mygenerator=generate_data(val_list,shape,batch_size)
#print('main',shape,batch_size)

for i,j in mygenerator:
    print(i)
    print('******************\n')
    print(j)
    print(np.shape(i),np.shape(j))
#rint(add(3,5))
'''
if __name__ == "__main__":
    print('main',shape,batch_size)
'''    
import os
import cv2
import numpy as np
import random
#from keras.preprocessing import image as Im
from datetime import datetime

read_dir='./images/'
write_dir='./aug_images/'

shape=(512,512)
'''
time=0
for filename in os.listdir(read_dir):
	if filename.endswith(".jpg"):
		#print(read_dir+filename)
		#image = cv2.resize(cv2.imread(read_dir+filename), shape)
		t1 = datetime.now()
		image=cv2.imread(read_dir+filename)
		t2 = datetime.now()
		#trans=np.random.choice(a=[True, False], size=(1,4))
		#print(trans)
		#if trans[0,0]:
		image=Im.random_rotation(image,rg=5, row_axis=0, col_axis=1, channel_axis=2,) 
		#if trans[0,1]:
		t3 = datetime.now()
		image=Im.random_shift(image, wrg=0.05, hrg=0.05, row_axis=0, col_axis=1, channel_axis=2)
		#if trans[0,2]:
		t4 = datetime.now()
		image=Im.random_brightness(image, brightness_range=(1-0.2,1+0.2))
		t5 = datetime.now()
		if bool(random.getrandbits(10)):
			image=Im.flip_axis(image,1)
		t6 = datetime.now()
		image=cv2.resize(image, shape)
		t7 = datetime.now()
		#cv2.imwrite(write_dir+filename,image)
		#t8 = datetime.now()

		time+=(t2-t1).microseconds+(t3-t2).microseconds+(t4-t3).microseconds+(t5-t4).microseconds+(t6-t5).microseconds+(t7-t6).microseconds
print('done',time)
'''
img_batch=[]
t1 = datetime.now()
for filename in os.listdir(read_dir):
	if filename.endswith(".jpg"):
		image=cv2.resize(cv2.imread(read_dir+filename), shape)
		if bool(random.getrandbits(1)):
			image=Im.flip_axis(image,1)
		img_batch.append(image)
		
		#cv2.imwrite(write_dir+filename,image)
img_batch=np.array(img_batch)
#print(np.shape(img_batch))

#img_batch=Im.random_rotation(img_batch,rg=5, row_axis=1, col_axis=2, channel_axis=3) 
#img_batch=Im.random_shift(img_batch, wrg=0.05, hrg=0.05, row_axis=1, col_axis=2, channel_axis=3)
#img_batch=Im.random_brightness(img_batch, brightness_range=(1-0.2,1+0.2))

img_batch=random_aug(img_batch, rg=5, wrg=0.05, hrg=0.05, row_axis=1, col_axis=2, channel_axis=3, fill_mode='nearest', cval=0.)
for i in img_batch[0]:
	cv2.imwrite(write_dir+filename,i)
t1 = datetime.now()
print('done',(t2-t1).microseconds)







'''
def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.random.uniform(-rg, rg)
    x = apply_affine_transform(x, theta=theta, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(x, tx=tx, ty=ty, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x
'''






def random_aug(x, rg, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.):
	theta = np.random.uniform(-rg, rg)
	h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(x, theta=theta, tx=tx, ty=ty, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x

def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0.):

    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if transform_matrix is not None:
    	for i in range(0,np.shape(x)[0]):
    		if bool(random.getrandbits(10)):
		        h, w = x[i].shape[row_axis], x[i].shape[col_axis]
		        transform_matrix = transform_matrix_offset_center(
		            transform_matrix, h, w)
		        x[i] = np.rollaxis(x[i], channel_axis, 0)
		        final_affine_matrix = transform_matrix[:2, :2]
		        final_offset = transform_matrix[:2, 2]

		        channel_images = [ndi.interpolation.affine_transform(
		            x_channel,
		            final_affine_matrix,
		            final_offset,
		            order=1,
		            mode=fill_mode,
		            cval=cval) for x_channel in x[i]]
		        x[i] = np.stack(channel_images, axis=0)
		        x[i] = np.rollaxis(x, 0, channel_axis + 1)
    return x
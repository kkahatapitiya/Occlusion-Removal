import os
import cv2
import numpy as np
import random
from keras.preprocessing import image as Im
from datetime import datetime
import scipy.ndimage as ndi

read_dir='./images/'
write_dir='./aug_images/'

shape=(512,512)

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
			print(np.shape(x[i]))
			y=x[i]
			if bool(random.getrandbits(1)):
				h, w = y.shape[row_axis], y.shape[col_axis]
				transform_matrix = Im.transform_matrix_offset_center(
				    transform_matrix, h, w)
				y = np.rollaxis(y, channel_axis-1, 0)
				final_affine_matrix = transform_matrix[:2, :2]
				final_offset = transform_matrix[:2, 2]

				channel_images = [ndi.interpolation.affine_transform(
				    x_channel,
				    final_affine_matrix,
				    final_offset,
				    order=1,
				    mode=fill_mode,
				    cval=cval) for x_channel in y]
				y = np.stack(channel_images, axis=0)
				x[i] = np.rollaxis(y, 0, channel_axis)
	return x


img_batch=[]
names=[]
t1 = datetime.now()
for filename in os.listdir(read_dir):
	if filename.endswith(".jpg"):
		names.append(filename)
		image=cv2.resize(cv2.imread(read_dir+filename), shape)
		if bool(random.getrandbits(1)):
			image=Im.flip_axis(image,1)
		img_batch.append(image)
		
		#cv2.imwrite(write_dir+filename,image)
img_batch=np.array(img_batch)
print(np.shape(img_batch))

#img_batch=Im.random_rotation(img_batch,rg=5, row_axis=1, col_axis=2, channel_axis=3) 
#img_batch=Im.random_shift(img_batch, wrg=0.05, hrg=0.05, row_axis=1, col_axis=2, channel_axis=3)
#img_batch=Im.random_brightness(img_batch, brightness_range=(1-0.2,1+0.2))

img_batch=random_aug(img_batch, rg=5, wrg=0.05, hrg=0.05, row_axis=1, col_axis=2, channel_axis=3, fill_mode='nearest', cval=0.)

t2 = datetime.now()
for i in range(0,np.shape(img_batch)[0]):
	print(write_dir+filename)
	cv2.imwrite(write_dir+names[i],img_batch[i])

print('done',(t2-t1).microseconds)
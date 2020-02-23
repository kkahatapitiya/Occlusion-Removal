import os
import numpy as np 
import cv2 as cv 
from PIL import Image

indir='./fake_images/in'
outdir='./fake_images/scaled'
scale=(600,600)

for file in os.listdir(indir):
	img = cv.imread(os.path.join(indir,file))
	res = cv.resize(img, scale)
	#res = np.resize(img, scale)
	#res = np.asarray(res)
	cv.imwrite(os.path.join(outdir,file),res)
	#saveimg = Image.fromarray(res)
	#saveimg.save(output)
from __future__ import print_function

import math
import os
import sys
import time
import json
import numpy as np
import cv2
import scipy.ndimage as ndi
import random
from six.moves import cPickle
from collections import deque
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing import image as Im
import keras.backend as K
import neuralgym as ng

segment_dir='./test_img/segment'
mask_dir='./test_img/mask'
sys.path.insert(0,'./models/')

from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import save_annotation
from deeplab import input_preprocess
from keras_resnet import resnet
from generative_inpainting.inpaint_model import InpaintCAModel

slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')
flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')
flags.DEFINE_string('checkpoint_deeplab', None, 'Directory of model checkpoints.')
flags.DEFINE_integer('vis_batch_size', 1,'The number of images in each batch during evaluation.')
flags.DEFINE_multi_integer('vis_crop_size', [513, 513],'Crop size [height, width] for visualization.')
flags.DEFINE_integer('eval_interval_secs', 60 * 5,'How often (in seconds) to run evaluation.')
flags.DEFINE_multi_integer('atrous_rates', None,'Atrous rates for atrous spatial pyramid pooling.')
flags.DEFINE_integer('output_stride', 16,'The ratio of input to output spatial resolution.')
flags.DEFINE_multi_float('eval_scales', [1.0],'The scales to resize images for evaluation.')
flags.DEFINE_bool('add_flipped_images', False,'Add flipped images for evaluation or not.')
flags.DEFINE_string('dataset', 'pascal_voc_seg','Name of the segmentation dataset.')
flags.DEFINE_string('vis_split', 'val','Which split of the dataset used for visualizing results')
flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')
flags.DEFINE_enum('colormap_type', 'pascal', ['pascal', 'cityscapes'],'Visualization colormap type.')
flags.DEFINE_boolean('also_save_raw_predictions', False,'Also save raw predictions.')
flags.DEFINE_integer('max_number_of_iterations', 0,'Maximum number of visualization iterations. Will loop ''indefinitely upon nonpositive values.')

flags.DEFINE_string('input_dir', None, 'Where is input image')
flags.DEFINE_string('output_dir', None, 'Where to write output.')
flags.DEFINE_string('checkpoint_gan', None, 'The directory of gan checkpoint.')
flags.DEFINE_string('checkpoint_resnet', None, 'The directory of keras_resnet checkpoint.')
flags.DEFINE_string('embeddings',None,'Where is embeddings')
flags.DEFINE_string('class_map_w2v',None,'Where is classmap')
flags.DEFINE_string('class_names',None,'Where is class list')


_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'
_IMAGE_FORMAT = '%06d_image'
_PREDICTION_FORMAT = '%06d_prediction'
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]

#for keras_resnet
shape=128 #32,64,128
classes=91
batch_size = 128 #32,64,128                                                                                                                               
nb_classes = classes                                                                                                                                      
nb_epoch = 200 #200                                                                                                                                       
data_augmentation = True #False 

#for w2v
hard_thres=0.1
#w2v_thres=0.5
similarity_thres=0.4 #0.3
#similarity_thres=0.7
thing_step=-1
stuff_step=91


class FullModel():
  def __init__(self,sess):
    self.sess=sess
    self.image_in=0
    self.filename=''
    self.distances={}
    self.to_mask=[]

    self.class_names=[i.split(':')[1][1:] for i in open(FLAGS.class_names).read().splitlines()]
    with open(FLAGS.embeddings) as f:
      self.class_embed = json.load(f)
    with open(FLAGS.class_map_w2v) as f:
      self.classmap = json.load(f)

    tf.logging.set_verbosity(tf.logging.INFO)
    self.dataset = segmentation_dataset.get_dataset(
        FLAGS.dataset, FLAGS.vis_split, dataset_dir=FLAGS.dataset_dir)
    #save_dir = FLAGS.vis_logdir+_SEMANTIC_PREDICTION_SAVE_FOLDER
    self.deeplab_restore_flag=True
    print('init for deeplab')

    self.img_rows, self.img_cols = shape, shape
    self.img_channels = 3
    self.model_2 = resnet.ResnetBuilder.build_resnet_101((self.img_channels, self.img_rows, self.img_cols), nb_classes)
    self.checkpoint_2 = ModelCheckpoint(FLAGS.checkpoint_resnet, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    if os.path.isfile(FLAGS.checkpoint_resnet):
      self.model_2.load_weights(FLAGS.checkpoint_resnet)
    self.model_2.compile(loss=self.multitask_loss,
                        optimizer='adam',
                        metrics=[self.mean_pred])
    print('init for keras resnet')

    self.model_3 = InpaintCAModel()
    self.gan_restore_flag=True
    print('init for gan\n')

  def multitask_loss(self, y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

  def mean_pred(self, y_true, y_pred):
    p=y_true; q=tf.cast((y_pred>0.1),tf.float32); 
    r=tf.multiply(p,q); s=K.sum(p)+K.sum(q)-2*K.sum(r)
    return K.sum(r)/(s+K.sum(r))

  def _process_batch(self, original_images, semantic_predictions, image_names,
                     image_heights, image_widths):
    (original_images,
     semantic_predictions,
     image_names,
     image_heights,
     image_widths) = self.sess.run([original_images, semantic_predictions,
                               image_names, image_heights, image_widths])
    #print semantic_predictions
    num_image = semantic_predictions.shape[0]
    for i in range(num_image):
      image_height = np.squeeze(image_heights[i])
      image_width = np.squeeze(image_widths[i])
      original_image = np.squeeze(original_images[i])
      semantic_prediction = np.squeeze(semantic_predictions[i])
      crop_semantic_prediction = semantic_prediction[:image_height, :image_width]
    return crop_semantic_prediction

  def predict_deeplab(self):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #+++++++++++++++++ For Deeplab +++++++++++++++++++++++++++#
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    print ("Deeplab starting...")
    #with tf.Graph().as_default(),tf.Session() as sess:
    ##################################################
    self.image=cv2.cvtColor(self.image_in,cv2.COLOR_BGR2RGB)
    self.label=np.zeros((np.shape(self.image)[0],np.shape(self.image)[1],1))
    self.image=tf.convert_to_tensor(self.image)
    self.label=tf.convert_to_tensor(self.label)
    self.imshape=np.shape(self.image)
    
    self.original_image, self.image, self.label = input_preprocess.preprocess_image_and_label(
        self.image,
        self.label,
        crop_height=FLAGS.vis_crop_size[0],
        crop_width=FLAGS.vis_crop_size[1],
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        min_scale_factor=1.,
        max_scale_factor=1.,
        scale_factor_step_size=0,
        ignore_label=self.dataset.ignore_label,
        is_training=False,
        model_variant=FLAGS.model_variant)

    self.imname=tf.convert_to_tensor('asdasdad') 
    self.imheight=tf.convert_to_tensor(self.imshape[0])
    self.imwidth=tf.convert_to_tensor(self.imshape[1])
    self.samples = {
        common.IMAGE: self.image,
        common.IMAGE_NAME: self.imname ,
        common.HEIGHT: self.imheight ,
        common.WIDTH: self.imwidth ,
        common.ORIGINAL_IMAGE: self.original_image
    }
    self.samples = tf.train.batch(
      self.samples,
      batch_size=FLAGS.vis_batch_size,
      num_threads=1,
      capacity=32 * FLAGS.vis_batch_size,
      allow_smaller_final_batch=True,
      dynamic_pad=True)
    ##################################################

    self.model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: self.dataset.num_classes},
        crop_size=FLAGS.vis_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)
  
    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      self.predictions = model.predict_labels(
    self.samples[common.IMAGE],
    model_options=self.model_options,
    image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      self.predictions = model.predict_labels_multi_scale(
    self.samples[common.IMAGE],
    model_options=self.model_options,
    eval_scales=FLAGS.eval_scales,
    add_flipped_images=FLAGS.add_flipped_images)
    self.predictions = self.predictions[common.OUTPUT_TYPE]
  
    if FLAGS.min_resize_value and FLAGS.max_resize_value:
      assert FLAGS.vis_batch_size == 1
  
      self.original_image = tf.squeeze(self.samples[common.ORIGINAL_IMAGE])
      self.original_image_shape = tf.shape(self.original_image)
      self.predictions = tf.slice(
    self.predictions,
    [0, 0, 0],
    [1, self.original_image_shape[0], self.original_image_shape[1]])
      self.resized_shape = tf.to_int32([tf.squeeze(self.samples[common.HEIGHT]),
           tf.squeeze(self.samples[common.WIDTH])])
      self.predictions = tf.squeeze(
    tf.image.resize_images(tf.expand_dims(self.predictions, 3),
               self.resized_shape,
               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
               align_corners=True), 3)
  
    #print("33")
  
    if self.deeplab_restore_flag:
      self.deeplab_restore_flag=False

      tf.train.get_or_create_global_step()

      self.vars1=['xception_65','image_pooling','aspp','concat_projection','decoder','logits','global_step']
      self.saver = tf.train.Saver([i for i in slim.get_variables_to_restore() if any(k in i.name for k in self.vars1)])

      self.last_checkpoint = None       
      #print("3")
      self.last_checkpoint = slim.evaluation.wait_for_new_checkpoint(
          FLAGS.checkpoint_deeplab, self.last_checkpoint)
      #start = time.time()
      tf.logging.info(
          'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                   time.gmtime()))
      tf.logging.info('Visualizing with model %s', self.last_checkpoint)


      self.saver.restore(self.sess, self.last_checkpoint)
      #print('xxx')
    
    self.coord=tf.train.Coordinator()
    tf.train.start_queue_runners(self.sess,coord=self.coord)

    tf.logging.info('Visualizing...')
    
    self.segmented_image=self._process_batch(
           original_images=self.samples[common.ORIGINAL_IMAGE],
           semantic_predictions=self.predictions,
           image_names=self.samples[common.IMAGE_NAME],
           image_heights=self.samples[common.HEIGHT],
           image_widths=self.samples[common.WIDTH],)

    #print('yyy')
    #cv2.imwrite(os.path.join('/Users/kumarakahatapitiya/Desktop/full_test/segment',filename),segmented_image)
    save_annotation.save_annotation(
        self.segmented_image, segment_dir,
        self.filename[:-4], add_colormap=True,
        colormap_type=FLAGS.colormap_type)
    tf.logging.info('Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',time.gmtime()))
    #print ("Finished visualizing...")
    #tf.reset_default_graph()

  def predict_kerasresnet(self):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #+++++++++++++++++ For Keras Resnet ++++++++++++++++++++++#
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    print ("Keras Resnet starting...")
    self.lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    self.early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    self.csv_logger = CSVLogger('resnet34_mscoco.csv')
    
    self.image=(cv2.resize(self.image_in, (shape,shape)).astype(float) - 128) / 128
    self.image=np.array([self.image])
    self.y_pred=self.model_2.predict(self.image, batch_size=1)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #+++++++++++++++++ For w2v +++++++++++++++++++++++++++++++#
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++# 

    col,cnt=np.unique(self.segmented_image,return_counts=True)
    cnt=cnt.astype('float32')
    print(np.shape(self.segmented_image))
    total=np.shape(self.segmented_image)[0]*np.shape(self.segmented_image)[1]
    
    lower_thres=0.02
    higher_thres=0.15
    #maxvalue_new=1.687
    #minvalue_new=0.667
    #cosine_min= -0.32078042674
    #cosine_max= 0.806910232728
    
    #for algo2
    cosine_min= -0.357849798828
    cosine_max= 0.81298648546
    k_size=5

    #print(cnt,col+thing_step,np.logical_and(cnt.astype('float')/total>lower_thres,cnt.astype('float')/total<higher_thres))
    self.things=np.ndarray.tolist(col[np.logical_and(cnt.astype('float')/total>lower_thres,cnt.astype('float')/total<higher_thres)]+thing_step)
    #self.things=np.ndarray.tolist(col+thing_step)
    #self.things.remove(thing_step)
    if thing_step in self.things: self.things.remove(thing_step)
    
    self.y_trunc=((self.y_pred>hard_thres)*1.0)[0] # 4864 vectors
    self.stuff=np.ndarray.tolist(np.nonzero(self.y_trunc)[0]+stuff_step)
    
    self.things_map=[(self.class_names[j],self.classmap[self.class_names[j]]['w2vclass']) for j in self.things] #classmap[j]['classnum']
    self.stuff_map=[(self.class_names[j],self.classmap[self.class_names[j]]['w2vclass']) for j in self.stuff]
    
    self.stuff_embed=[]
    for j in self.stuff_map:
      self.stuff_embed.append(self.class_embed[j[1]])
    
    if len(self.things)==0:
      print("No things\n")
    elif len(self.stuff_embed)==0:
      print("No stuff\n")
    else:
      '''
      #algo1
      self.stuff_embed=np.array(self.stuff_embed,dtype=np.float32)
      self.centroid=np.mean(self.stuff_embed,axis=0)    
      for k in self.things_map:
        self.dist=float("{0:.3f}".format(np.dot(self.class_embed[k[1]],self.centroid)))
        self.dist=(np.dot(self.class_embed[k[1]],self.centroid)-cosine_min)/(cosine_max-cosine_min)
        #self.dist=float("{0:.3f}".format((np.linalg.norm((self.class_embed[k[1]]-self.centroid),ord=2)-minvalue_new)/(maxvalue_new-minvalue_new)))
        self.distances[k[0]]=self.dist
      '''
      #algo2
      for k in self.things_map:
        self.others=self.stuff_map+self.things_map; self.others.remove(k); self.dist=0;
        for j in self.others:
          self.dist+=(np.dot(self.class_embed[k[1]],self.class_embed[j[1]])-cosine_min)/(cosine_max-cosine_min)
        self.distances[k[0]]=self.dist/len(self.others)
      
      self.similarity=[(l,list(self.distances.keys())[list(self.distances.values()).index(l)]) for l in (sorted(list(self.distances.values())))]
      
      #print(self.stuff_map)
      #print(self.similarity)
      
      #create mask for gan
      for l in self.similarity:
        if l[0]<similarity_thres:
        #if l[0]>similarity_thres:
          self.to_mask.append(self.class_names.index(l[1]))
      
      self.mask=np.zeros((np.shape(self.segmented_image)[0],np.shape(self.segmented_image)[1],3))
      if (self.to_mask)!=0:
        for i in self.to_mask:
          self.mask[np.where(self.segmented_image == i-thing_step)]=[255,255,255]
      #print ("Mask generated...")

      self.kernel = np.ones((k_size,k_size), np.uint8)
      self.mask = cv2.dilate(self.mask, self.kernel, iterations=3)

      cv2.imwrite(os.path.join(mask_dir,self.filename),self.mask)

  def predict_gan(self):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #+++++++++++++++++ For gan +++++++++++++++++++++++++++++++#
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if len(self.things)!=0 and len(self.stuff_embed)!=0:
      print ("GAN starting...")
      #with tf.Graph().as_default(),tf.Session() as sess:
        #model_3 = InpaintCAModel()
      
      self.image=self.image_in
      #print(image.shape,mask.shape)
      assert self.image.shape == self.mask.shape
      self.h, self.w, _ = self.image.shape
      self.grid = 8
      self.image = self.image[:self.h//self.grid*self.grid, :self.w//self.grid*self.grid, :]
      self.mask = self.mask[:self.h//self.grid*self.grid, :self.w//self.grid*self.grid, :]
      #print('Shape of image: {}'.format(image.shape))
      
      self.image = np.expand_dims(self.image, 0)
      self.mask = np.expand_dims(self.mask, 0)
      self.input_image = np.concatenate([self.image, self.mask], axis=2)
      
      self.input_image = tf.constant(self.input_image, dtype=tf.float32)
      self.output = self.model_3.build_server_graph(self.input_image,reuse=tf.AUTO_REUSE) #reuse=tf.AUTO_REUSE
    

      if self.gan_restore_flag:
        self.gan_restore_flag=False
        # load pretrained model
        self.vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars_list=[i for i in self.vars_list if 'inpaint_net' in i.name]
        #print(vars_list)
        self.assign_ops = []
        for var in self.vars_list:
            self.vname = var.name
            self.from_name = self.vname
            self.var_value = tf.contrib.framework.load_variable(FLAGS.checkpoint_gan, self.from_name)
            self.assign_ops.append(tf.assign(var, self.var_value))
        
        self.sess.run(self.assign_ops)
        #print('gan model loaded.')
        
      self.output = (self.output + 1.) * 127.5
      self.output = tf.reverse(self.output, [-1])
      self.output = tf.saturate_cast(self.output, tf.uint8)
      self.result = self.sess.run(self.output)
      cv2.imwrite(os.path.join(FLAGS.output_dir,self.filename), self.result[0][:, :, ::-1])
      print ("Successful finish\n")

    tf.get_variable_scope().reuse_variables()
      #tf.get_variable_scope().reuse_variables() #.reuse == True #scope.reuse_variables()
        #print(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
        #print(sess.run(tf.contrib.memory_stats.BytesInUse()))
      #tf.reset_default_graph()

def main(unused_argv):
  with tf.Session() as sess:
    fullmodel=FullModel(sess)

    for filename in os.listdir(FLAGS.input_dir):
      if filename.endswith('.jpg') or filename.endswith('.png'):
        
        print("Loading image: ",filename)
        fullmodel.image_in=cv2.imread(os.path.join(FLAGS.input_dir,filename))
        fullmodel.distances={}
        fullmodel.to_mask=[]
        fullmodel.filename=filename

        fullmodel.predict_deeplab()
        fullmodel.predict_kerasresnet()
        fullmodel.predict_gan()




if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_deeplab')
  flags.mark_flag_as_required('vis_logdir')
  flags.mark_flag_as_required('dataset_dir')
  flags.mark_flag_as_required('output_dir')
  flags.mark_flag_as_required('checkpoint_gan')
  flags.mark_flag_as_required('checkpoint_resnet')
  flags.mark_flag_as_required('input_dir')
  flags.mark_flag_as_required('embeddings')
  flags.mark_flag_as_required('class_map_w2v')
  flags.mark_flag_as_required('class_names')
  tf.app.run()

# Holy resources:
# https://github.com/virafpatrawala/COCO-Semantic-Segmentation/blob/master/COCOdataset_SemanticSegmentation_Demo.ipynb
# https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/
# https://blog.paperspace.com/data-augmentation-for-object-detection-building-input-pipelines/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import os
import tensorflow as tf
import skimage.io as io
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):

  def __init__(self, csv_file, base_dir, output_size, shuffle=False, batch_size=10):
    self.df = pd.read_csv(csv_file) # csv containing df of the chosen annotations
    self.base_dir = base_dir        # where to read imgs from in collab runtime
    self.output_size = output_size
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.on_epoch_end()

  # after each epoch, shuffle indices so data order changes
  def on_epoch_end(self):
    self.indices = np.arange(len(self.df))
    if self.shuffle:
      np.random.shuffle(self.indices)

  # number of batches (not number of examples)
  def __len__(self):
    return int(len(self.df) / self.batch_size)

  # scale RGB channels to [0,1]
  def scale_input(self,img):
    return img/255.0 

  # resize img to input shape (will likely replace once we start dealing with bboxes)
  def resize_img(self,img):
    return cv2.resize(img,self.output_size,interpolation=cv2.INTER_LINEAR) # what interpolation is best?

  # adjust x/y coords to new resized img
  def resize_label(self,label,og_width,og_height):
    x_scale = self.output_size[0]/og_width
    y_scale = self.output_size[1]/og_height
    return [int(round(int(val)*(x_scale if i%3==0 else y_scale if i%3==1 else 1))) for i,val in enumerate(label)]

  # returns batch at index idx
  def __getitem__(self, idx):
    ## Initialize Batch:
    X = np.empty((self.batch_size, *self.output_size, 3)) # 3 channels of RGB
    y = np.empty((self.batch_size, 3 * 17)) # (x,y,v) * 17 keypoints

    # get the indices of the requested batch
    indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

    for i, data_index in enumerate(indices):
      # Load label and img
      ann = self.df.loc[data_index]
      img = mpimg.imread(os.path.join(self.base_dir,ann['path'])) # bottleneck

      # Preprocess img
      img = DataGenerator.scale_input(self,img)
      og_width, og_height = int(img.shape[1]), int(img.shape[0])
      img = DataGenerator.resize_img(self,img)
      
      # Preprocess label
      label = ann['keypoints'].strip('][').split(', ')
      label = DataGenerator.resize_label(self,label,og_width,og_height)

      # Accumulate
      X[i,] = img
      y[i,] = np.asarray(label)
    return X, y

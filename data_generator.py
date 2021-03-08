# Holy resources:
# https://github.com/virafpatrawala/COCO-Semantic-Segmentation/blob/master/COCOdataset_SemanticSegmentation_Demo.ipynb
# https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/
# https://blog.paperspace.com/data-augmentation-for-object-detection-building-input-pipelines/

import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from constants import DEFAULT_BATCH_SIZE
from constants import NUM_COCO_KEYPOINTS
from constants import NUM_COCO_KP_ATTRBS
from constants import OUTPUT_DIM

class DataGenerator(Sequence):

  def __init__(self, csv_file, base_dir, output_size, shuffle=False, batch_size=DEFAULT_BATCH_SIZE):
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

  def transform_image(self, img):
    og_width, og_height = int(img.shape[1]), int(img.shape[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # fix RBG so images are easier to plot
    img = img/255.0 # scale RGB channels to [0,1]
    img = cv2.resize(img,self.output_size,interpolation=cv2.INTER_LINEAR) # resize
    return img, og_width, og_height
  
  def transform_label(self,label, og_width, og_height):
    label = label.strip('][').split(', ')
    x_scale = self.output_size[0]/og_width
    y_scale = self.output_size[1]/og_height
    # adjust x/y coords to new resized img
    transformed_label = [int(round(int(val)*(x_scale if i%NUM_COCO_KP_ATTRBS==0 else y_scale if i%NUM_COCO_KP_ATTRBS==1 else 1))) for i,val in enumerate(label)]
    return np.asarray(transformed_label)

  # returns batch at index idx
  def __getitem__(self, idx):
    # Initialize Batch:
    X = np.empty((self.batch_size, *self.output_size, 3)) # 3 channels of RGB
    y = np.empty((self.batch_size, NUM_COCO_KP_ATTRBS * NUM_COCO_KEYPOINTS))

    # get the indices of the requested batch
    indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
    for i, data_index in enumerate(indices):
      ann = self.df.loc[data_index]
      img_path = os.path.join(self.base_dir,ann['path'])
      img = cv2.imread(img_path) # bottleneck

      transformed_img, og_width, og_height = self.transform_image(img)
      transformed_label = self.transform_label(ann['keypoints'],og_width,og_height)

      X[i,] = transformed_img
      y[i,] = transformed_label
    return X, y
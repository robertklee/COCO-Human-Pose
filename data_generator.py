# Holy resources:
# https://github.com/virafpatrawala/COCO-Semantic-Segmentation/blob/master/COCOdataset_SemanticSegmentation_Demo.ipynb
# https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/
# https://blog.paperspace.com/data-augmentation-for-object-detection-building-input-pipelines/

import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
import skimage.io as io
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

  def transform_image(self, img, bbox):
    new_bbox = self.transform_bbox(bbox)
    cropped_img = img.crop(box=new_bbox)
    cropped_width, cropped_height = cropped_img.size
    img = np.array(cropped_img)
    img = img/255.0 # scale RGB channels to [0,1]
    img = cv2.resize(img,self.output_size,interpolation=cv2.INTER_LINEAR) # resize
    return img, cropped_width, cropped_height, new_bbox[0], new_bbox[1]
  
  def transform_bbox(self,bbox):
    x,y,w,h = [int(round(i)) for i in list(map(float,(bbox.strip('][').split(', '))))] # (x,y,w,h) anchored to top left
    center_x = int(x+w/2)
    center_y = int(y+h/2)
    new_w = w if w >= h else int(h * self.output_size[0]/self.output_size[1])
    new_h = h if w <  h else int(w * self.output_size[1]/self.output_size[0])
    new_x = int(center_x - new_w/2)
    new_y = int(center_y - new_h/2)
    return (new_x,new_y,new_x+new_w,new_y+new_h)

  def transform_label(self,label, cropped_width, cropped_height,anchor_x,anchor_y):
    label = [int(v) for v in label.strip('][').split(', ')]
    # adjust x/y coords to new resized img
    transformed_label = []
    for x, y, v in zip(*[iter(label)]*3):
      x = (x-anchor_x) * self.output_size[0]/cropped_width
      y = (y-anchor_y) * self.output_size[1]/cropped_height

      # validate kps, throw away if out of bounds
      # TODO: if kp is thrown away then we must update num_keypoints
      if (x > self.output_size[0] or x < 0) or (y > self.output_size[1] or y < 0):
        x,y,v = (0,0,0)
      
      transformed_label.append(x)
      transformed_label.append(y)
      transformed_label.append(v)
    
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
      
      #img = Image.open(img_path) # bottleneck opening from file system
      img = Image.fromarray(io.imread(ann['coco_url'])) # bottleneck opening from URL
      transformed_img, cropped_width, cropped_height, anchor_x, anchor_y = self.transform_image(img, ann['bbox'])
      transformed_label = self.transform_label(ann['keypoints'],cropped_width,cropped_height,anchor_x,anchor_y)

      X[i,] = transformed_img
      y[i,] = transformed_label
    return X, y
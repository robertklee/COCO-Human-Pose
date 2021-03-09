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

class DataGenerator(Sequence): # inherit from Sequence to access multicore functionality: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

  def __init__(self, csv_file, base_dir, input_dim, output_dim, num_hg_blocks, shuffle=False, batch_size=DEFAULT_BATCH_SIZE):
    self.df = pd.read_csv(csv_file) # csv with df of the the annotations we want, we may eventually want to pass in df instead
    self.base_dir = base_dir        # where to read imgs from in collab runtime
    self.input_dim = input_dim      # model requirement for input image dimensions
    self.output_dim = output_dim    # dimesnions of output heatmap of model
    self.num_hg_blocks = num_hg_blocks
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
    scaled_img = np.array(cropped_img)/255.0 # scale RGB channels to [0,1]
    new_img = cv2.resize(scaled_img,self.input_dim,interpolation=cv2.INTER_LINEAR)
    return new_img, cropped_width, cropped_height, new_bbox[0], new_bbox[1]
  
  def transform_bbox(self,bbox):
    x,y,w,h = [round(i) for i in list(map(float,(bbox.strip('][').split(', '))))] # (x,y,w,h) anchored to top left
    center_x = int(x+w/2)
    center_y = int(y+h/2)
    new_w = w if w >= h else round(h * self.input_dim[0]/self.input_dim[1])
    new_h = h if w <  h else round(w * self.input_dim[1]/self.input_dim[0])
    new_x = int(center_x - new_w/2)
    new_y = int(center_y - new_h/2)
    return (new_x,new_y,new_x+new_w,new_y+new_h)

  def transform_label(self,label, cropped_width, cropped_height,anchor_x,anchor_y):
    label = [int(v) for v in label.strip('][').split(', ')]
    # adjust x/y coords to new resized img
    transformed_label = []
    for x, y, v in zip(*[iter(label)]*3):
      x = round((x-anchor_x) * self.input_dim[0]/cropped_width)
      y = round((y-anchor_y) * self.input_dim[1]/cropped_height)
      # validate kps, throw away if out of bounds
      # TODO: if kp is thrown away then we must update num_keypoints
      if (x > self.input_dim[0] or x < 0) or (y > self.input_dim[1] or y < 0):
        x,y,v = (0,0,0)
      
      transformed_label.append(x)
      transformed_label.append(y)
      transformed_label.append(v)
    return np.asarray(transformed_label)

  def generate_heatmaps(self,label):
    heat_maps = np.zeros((*self.output_dim, self.num_hg_blocks * NUM_COCO_KEYPOINTS))
    for i in range(NUM_COCO_KEYPOINTS):
      label_idx = i * NUM_COCO_KP_ATTRBS
      heatmap_idx = i * self.num_hg_blocks

      if label[label_idx + (NUM_COCO_KP_ATTRBS-1)] == 0: # generate empty heatmap for unlabelled kp
        continue
      kpx = int(label[label_idx] * (self.output_dim[0]/self.input_dim[0]))     # How should kp coords be translated from 256*256 to 64*64, we lose precision here
      kpy = int(label[label_idx + 1] * (self.output_dim[1]/self.input_dim[1])) #  this loss of precision results in clouds not perfectly centered around gt kp
      heat_map = self.gaussian(heat_maps[:,:,heatmap_idx], (kpx,kpy),2) # what should sigma be?
      heat_maps[:,:,heatmap_idx:heatmap_idx+self.num_hg_blocks] = np.repeat(heat_map[:,:,np.newaxis],repeats=self.num_hg_blocks,axis=2) 
    return heat_maps

  # This func is unmodified and ripped from: https://github.com/princeton-vl/pose-hg-train/blob/master/src/pypose/draw.py
  def gaussian(self,img, pt, sigma):
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img
 

  # returns batch at index idx
  def __getitem__(self, idx):
    # Initialize Batch:
    X = np.empty((self.batch_size, *self.input_dim, 3)) # 3 channels of RGB
    y = np.empty((self.batch_size, *self.output_dim, self.num_hg_blocks * NUM_COCO_KEYPOINTS))

    # get the indices of the requested batch
    indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
    for i, data_index in enumerate(indices):
      ann = self.df.loc[data_index]
      img_path = os.path.join(self.base_dir,ann['path'])
      
      img = Image.open(img_path) # bottleneck opening from file system
      #img = Image.fromarray(io.imread(ann['coco_url'])) # bottleneck opening from URL

      transformed_img, cropped_width, cropped_height, anchor_x, anchor_y = self.transform_image(img, ann['bbox'])
      transformed_label = self.transform_label(ann['keypoints'],cropped_width,cropped_height,anchor_x,anchor_y)
      heat_map_labels = self.generate_heatmaps(transformed_label)
      X[i,] = transformed_img
      y[i,] = heat_map_labels
    return X, y
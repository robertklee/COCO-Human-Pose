# Holy resources:
# https://github.com/virafpatrawala/COCO-Semantic-Segmentation/blob/master/COCOdataset_SemanticSegmentation_Demo.ipynb
# https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/
# https://blog.paperspace.com/data-augmentation-for-object-detection-building-input-pipelines/

import os

import cv2
import numpy as np
import pandas as pd
import skimage.io as io
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import Sequence

from constants import *

# TODO: Data Augementation:
# - bounding box varied through data augmentation 110% to 150%
# - horizontal flips
# - rotations
# - brightness adjustments
# - contrast
# - noise/grain


# inherit from Sequence to access multicore functionality: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):

    def __init__(self, df, base_dir, input_dim, output_dim, num_hg_blocks, shuffle=False, batch_size=DEFAULT_BATCH_SIZE, online_fetch=False):
        self.df = df                    # df of the the annotations we want
        self.base_dir = base_dir        # where to read imgs from in collab runtime
        self.input_dim = input_dim      # model requirement for input image dimensions
        self.output_dim = output_dim    # dimesnions of output heatmap of model
        self.num_hg_blocks = num_hg_blocks
        self.shuffle = shuffle
        self.batch_size = batch_size
        # If true, images will be loaded from url over network rather than filesystem
        self.online_fetch = online_fetch
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
        scaled_img = np.array(cropped_img)/255.0  # scale RGB channels to [0,1]
        new_img = cv2.resize(scaled_img, self.input_dim,
                             interpolation=cv2.INTER_LINEAR)
        return new_img, cropped_width, cropped_height, new_bbox[0], new_bbox[1]

    def transform_bbox(self, bbox):
        x, y, w, h = [i for i in bbox]  # (x,y,w,h) anchored to top left
        center_x = x+w/2
        center_y = y+h/2
        if w >= h:
            new_w = w
        else:
            new_w = h * self.input_dim[0]/self.input_dim[1]
        if w < h:
            new_h = h
        else:
            new_h = w * self.input_dim[1]/self.input_dim[0]
        new_w *= BBOX_SLACK  # add slack to bbox
        new_h *= BBOX_SLACK  # add slack to bbox
        new_x = center_x - new_w/2
        new_y = center_y - new_h/2
        return (round(new_x), round(new_y), round(new_x+new_w), round(new_y+new_h))

    def transform_label(self, label, cropped_width, cropped_height, anchor_x, anchor_y):
        label = [int(v) for v in label]
        # adjust x/y coords to new resized img
        transformed_label = []
        for x, y, v in zip(*[iter(label)]*NUM_COCO_KP_ATTRBS):
            x = round((x-anchor_x) * self.input_dim[0]/cropped_width)
            y = round((y-anchor_y) * self.input_dim[1]/cropped_height)
            # validate kps, throw away if out of bounds
            # TODO: if kp is thrown away then we must update num_keypoints
            if (x > self.input_dim[0] or x < 0) or (y > self.input_dim[1] or y < 0):
                x, y, v = (0, 0, 0)

            transformed_label.append(x)
            transformed_label.append(y)
            transformed_label.append(v)
        return np.asarray(transformed_label)

    def generate_heatmaps(self, label):
        heat_maps = np.zeros((*self.output_dim, NUM_COCO_KEYPOINTS))
        for i in range(NUM_COCO_KEYPOINTS):
            label_idx = i * NUM_COCO_KP_ATTRBS  # index for label
            # generate empty heatmap for unlabelled kp
            if label[label_idx + (NUM_COCO_KP_ATTRBS-1)] == 0:
                continue
            kpx = int(label[label_idx])
            kpy = int(label[label_idx + 1])
            heat_map = self.gaussian(
                np.zeros(self.input_dim), (kpx, kpy), HEATMAP_SIGMA)
            # downscale heatmap resolution
            heat_maps[:, :, i] = cv2.resize(
                heat_map, self.output_dim, interpolation=cv2.INTER_LINEAR)

        return heat_maps

    # This func is unmodified and ripped from: https://github.com/princeton-vl/pose-hg-train/blob/master/src/pypose/draw.py
    def gaussian(self, img, pt, sigma):
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

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]
            ] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img

    # returns batch at index idx

    def __getitem__(self, idx):
        # Initialize Batch:
        X = np.empty((self.batch_size, *self.input_dim, 3))

        # Order of last dimension: (heatmap for each kp) repeated num_hg_blocks times
        y = np.empty((self.batch_size, *self.output_dim, NUM_COCO_KEYPOINTS))

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        for i, data_index in enumerate(indices):
            ann = self.df.loc[data_index]
            img_path = os.path.join(self.base_dir, ann['path'])

            if self.online_fetch:
                img = Image.fromarray(io.imread(ann['coco_url'])).convert(
                    'RGB')  # bottleneck opening from URL
            else:
                # bottleneck opening from file system
                img = Image.open(img_path).convert('RGB')

            transformed_img, cropped_width, cropped_height, anchor_x, anchor_y = self.transform_image(
                img, ann['bbox'])
            transformed_label = self.transform_label(
                ann['keypoints'], cropped_width, cropped_height, anchor_x, anchor_y)
            heat_map_labels = self.generate_heatmaps(transformed_label)
            X[i, ] = transformed_img
            y[i, ] = heat_map_labels

            y_stacked = []
            for j in range(self.num_hg_blocks):
                y_stacked.append(y)

        return X, y_stacked

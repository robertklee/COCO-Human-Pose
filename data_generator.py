# Holy resources:
# https://github.com/virafpatrawala/COCO-Semantic-Segmentation/blob/master/COCOdataset_SemanticSegmentation_Demo.ipynb
# https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/
# https://blog.paperspace.com/data-augmentation-for-object-detection-building-input-pipelines/

import os

import cv2
import imgaug as ia
import numpy as np
import pandas as pd
import skimage.io as io
import tensorflow as tf
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from PIL import Image
from tensorflow.keras.utils import Sequence

from constants import *
from data_augmentation import *


def transform_bbox_square(bbox, slack=1):
    """
        Transforms a bounding box anchored at top left corner of shape () to a square with
        edge length being the larger of the bounding box's height or width.

        Only supports square aspect ratios currently.

        ## Parameters

        bbox : {tuple or ndarray of len 4}
            Given as two points, anchored at top left of image being 0,0: left, upper, right, lower

        slack : {int, float}
            The amount of extra padding that should be applied to the edges of the bounding box after
            transforming
        ##
    """
    x, y, w, h = [i for i in bbox]  # (x,y,w,h) anchored to top left
    center_x = x+w/2
    center_y = y+h/2

    if w >= h:
        new_w = w
        new_h = w
    else:
        new_w = h
        new_h = h

    new_w *= slack  # add slack to bbox
    new_h *= slack  # add slack to bbox
    new_x = center_x - new_w/2
    new_y = center_y - new_h/2
    return (round(new_x), round(new_y), round(new_x+new_w), round(new_y+new_h))

# inherit from Sequence to access multicore functionality: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):

    def __init__(self, df, base_dir, input_dim, output_dim, num_hg_blocks, shuffle=False, \
        batch_size=DEFAULT_BATCH_SIZE, online_fetch=False, img_aug_strength=None, is_eval=False):

        self.df = df                    # df of the the annotations we want
        self.base_dir = base_dir        # where to read imgs from in collab runtime
        # NOTE update image transformation logic if input is no longer square
        self.input_dim = input_dim      # model requirement for input image dimensions
        self.output_dim = output_dim    # dimesnions of output heatmap of model
        self.num_hg_blocks = num_hg_blocks
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.is_eval = is_eval
        # If true, images will be loaded from url over network rather than filesystem
        self.online_fetch = online_fetch
        if img_aug_strength is not None:
            self.augmenter = get_augmenter_pipeline(img_aug_strength)
        else:
            self.augmenter = None

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
        new_bbox = transform_bbox_square(bbox, slack=BBOX_SLACK)
        cropped_img = img.crop(box=new_bbox)
        cropped_width, cropped_height = cropped_img.size
        new_img = cv2.resize(np.array(cropped_img), self.input_dim,
                             interpolation=cv2.INTER_LINEAR)
        return new_img, cropped_width, cropped_height, new_bbox[0], new_bbox[1]

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
                heat_map, self.output_dim, interpolation=cv2.INTER_AREA)
            heat_maps[:, :, i] /= (heat_maps[:, :, i].max()) # normalize heatmap to [0,1]
            heat_maps[:, :, i] *= HEATMAP_SCALE # scale up to place more importance on correctly identifying kp regions

        return heat_maps

    def convert_coco_kp_to_imgaug_kp(self, label):
        kps = []
        valid = np.ones(NUM_COCO_KEYPOINTS)
        invalid_xy = -1
        for i in range(NUM_COCO_KEYPOINTS):
            label_idx = i * NUM_COCO_KP_ATTRBS  # index for label
            # generate empty heatmap for unlabelled kp
            if label[label_idx + (NUM_COCO_KP_ATTRBS-1)] == 0:
                # invalid keypoint
                valid[i] = 0
                kps.append(Keypoint(x=invalid_xy, y=invalid_xy))
                continue
            kpx = int(label[label_idx])
            kpy = int(label[label_idx + 1])
            kps.append(Keypoint(x=kpx, y=kpy))

        return kps, valid

    def convert_imgaug_kpsoi_to_coco_kp(self, kpsoi_aug, valid, image_aug):
        transformed_label = []

        for i in range(NUM_COCO_KEYPOINTS):
            kp = kpsoi_aug[i]
            if (not valid[i]) or kp.is_out_of_image(image_aug):
                x, y, v = (0, 0, 0)
            else:
                x = kp.x
                y = kp.y
                v = 1

            transformed_label.append(x)
            transformed_label.append(y)
            transformed_label.append(v)


        return np.asarray(transformed_label)

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

    """
    Returns a batch from the dataset

    ### Parameters:
    idx : {int-type} Batch number to retrieve

    ### Returns:
    Tuple of (X, y) where:

    X : ndarray of shape (batch number, input_dim1, input_dim2, 3)
        This corresponds to a batch of images, normalized from [0,255] to [0,1]

    y : list of ndarrays where each list element corresponds to an intermediate (or final) layer of the hourglass,
        and has shape (batch number, output_dim1, output_dim2, 17). The list length is num_hg_blocks

        Each output corresponds to a heatmap, which currently is a Gaussian and has range [0,1]
    """
    def __getitem__(self, idx):
        # Initialize Batch:
        X = np.empty((self.batch_size, *self.input_dim, INPUT_CHANNELS))

        # Order of last dimension: (heatmap for each kp) repeated num_hg_blocks times
        y = np.empty((self.batch_size, *self.output_dim, NUM_COCO_KEYPOINTS))

        metadatas = []
        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        for i, data_index in enumerate(indices):
            ann = self.df.loc[data_index]
            img_path = os.path.join(self.base_dir, ann['path'])

            if self.online_fetch:
                img = Image.fromarray(io.imread(ann['coco_url'])).convert('RGB')  # bottleneck opening from URL
            else:
                # bottleneck opening from file system
                img = Image.open(img_path).convert('RGB')

            transformed_img, cropped_width, cropped_height, anchor_x, anchor_y = self.transform_image(
                img, ann['bbox'])
            transformed_label = self.transform_label(
                ann['keypoints'], cropped_width, cropped_height, anchor_x, anchor_y)

            if self.is_eval:
                metadata = {}
                metadata["src_set_image_id"] = ann['src_set_image_id']
                metadata["ann_id"] = ann['ann_id']
                metadata["coco_url"] = ann['coco_url']
                metadata["cropped_width"] = cropped_width
                metadata["cropped_height"] = cropped_height
                metadata["anchor_x"] = anchor_x
                metadata["anchor_y"] = anchor_y
                metadata["input_dim"] = self.input_dim
                metadata["output_dim"] = self.output_dim
                metadata["transformed_label"] = transformed_label #DEBUG
                metadata["ground_truth_keypoints"] = ann['keypoints'] #DEBUG
                metadatas.append(metadata)

            # if image augmentations should be applied
            if self.augmenter is not None:
                imgaug_kps, valid = self.convert_coco_kp_to_imgaug_kp(transformed_label.astype('float32'))

                # Keep track of image dimension
                kpsoi = KeypointsOnImage(imgaug_kps, shape=transformed_img.shape)

                # Perform data augmentation randomly
                image_aug, kpsoi_aug = self.augmenter(image=transformed_img, keypoints=kpsoi)

                # Perform a R/L augmentation randomly, applying R/L flip to the labels as well to maintain the right order 
                image_aug, kpsoi_aug = flipRL(image=image_aug,keypoints=kpsoi_aug)

                # Filter out out-of-bounds (from rotation/cropping) and invalid (originally occluded/not present) keypoints
                augmented_label = self.convert_imgaug_kpsoi_to_coco_kp(kpsoi_aug, valid, image_aug)

                # Update data
                transformed_img = image_aug
                transformed_label = augmented_label

            normalized_img = transformed_img/255.0  # scale RGB channels to [0,1]

            heat_map_labels = self.generate_heatmaps(transformed_label)
            X[i, ] = normalized_img
            y[i, ] = heat_map_labels

        y_stacked = []
        for _ in range(self.num_hg_blocks):
            y_stacked.append(y)

        if self.is_eval:
            return X, y_stacked, metadatas

        return X, y_stacked

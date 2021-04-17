import csv
import imghdr
import io
import json
import os
import re
from functools import lru_cache

import cv2
import keras
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from keras.models import load_model, model_from_json
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, maximum_filter

import data_generator
import evaluation
import util
from constants import *


class AppHelper():
    def __init__(self, model_weights, model_json) -> None:
        self._load_model(model_json=model_json, model_weights=model_weights)

    def predict_in_memory(self, img, visualize_scatter=True, visualize_skeleton=True, average_flip_prediction=True):
        # Number of hg blocks doesn't matter
        X_batch, y_stacked = evaluation.load_and_preprocess_img(img, 1)
        y_batch = y_stacked[0] # take first hourglass section
        img_id_batch = None

        return self._predict_and_visualize(
            X_batch,
            visualize_scatter=visualize_scatter,
            visualize_skeleton=visualize_skeleton,
            average_flip_prediction=average_flip_prediction
        )

    def _load_model(self, model_json, model_weights):
        with open(model_json) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(model_weights)

    def _predict_and_visualize(self, X_batch, visualize_scatter=True, visualize_skeleton=True, average_flip_prediction=True):
        predicted_heatmaps_batch = self.predict_heatmaps(X_batch)

        if visualize_scatter or visualize_skeleton:
            # Get predicted keypoints from last hourglass (last element of list)
            # Dimensions are (hourglass_layer, batch, x, y, keypoint)
            keypoints_batch = self.heatmaps_to_keypoints_batch(predicted_heatmaps_batch)

            if average_flip_prediction:
                # Average predictions from original image and the untransformed flipped image to get a more accurate prediction
                predicted_heatmaps_batch_2 = self.predict_heatmaps(X_batch=X_batch, predict_using_flip=True)

                keypoints_batch_2 = self.heatmaps_to_keypoints_batch(predicted_heatmaps_batch_2)

                for i in range(keypoints_batch.shape[0]):
                    # Average predictions from normal and flipped input
                    keypoints_batch[i] = self._average_LR_flip_predictions(keypoints_batch[i], keypoints_batch_2[i], coco_format=False)

            # if visualize_skeleton:
            #     # Plot only skeleton
            #     self.visualize_keypoints(np.zeros(X_batch.shape), keypoints_batch, show_skeleton=visualize_skeleton)

            # Plot skeleton with image
            return self.visualize_keypoints(X_batch, keypoints_batch, show_skeleton=visualize_skeleton)


    def _average_LR_flip_predictions(self, prediction_1, prediction_2, coco_format=True):
        # Average predictions from original image and the untransformed flipped image to get a more accurate prediction
        original_shape = prediction_1.shape

        prediction_1_flat = prediction_1.flatten()
        prediction_2_flat = prediction_2.flatten()

        output_prediction = prediction_1_flat

        for j in range(NUM_COCO_KEYPOINTS):
            # This code is required so if one version detects the keypoint (x,y,1),
            # and the other doesn't (0,0,0), we don't average them to be (x/2, y/2, 0.5)
            base = j * NUM_COCO_KP_ATTRBS

            n = 0
            x_sum = 0
            y_sum = 0
            vc_sum = 0 # Could be visibility or confidence

            # Verify visibility flag
            if prediction_1_flat[base+2] >= HM_TO_KP_THRESHOLD:
                x_sum += prediction_1_flat[base]
                y_sum += prediction_1_flat[base + 1]
                vc_sum += prediction_1_flat[base + 2]
                n += 1

            if prediction_2_flat[base+2] >= HM_TO_KP_THRESHOLD:
                x_sum += prediction_2_flat[base]
                y_sum += prediction_2_flat[base + 1]
                vc_sum += prediction_2_flat[base + 2]
                n += 1

            # Verify that no division by 0 will occur
            if n > 0:
                output_prediction[base]     = round(x_sum / n)
                output_prediction[base + 1] = round(y_sum / n)
                output_prediction[base + 2] = 1 if coco_format else round(vc_sum / n)

            ## There is probably some numpy method to do this. The following line doesn't work because it doesn't account for the vis flag being 0,
            ## which causes the x,y to be (0,0)
            # list_of_predictions[i]['keypoints'] = np.round(np.mean( np.array([ list_of_predictions[i]['keypoints'], list_of_predictions_2[i]['keypoints'] ]), axis=0 ))

        if not coco_format:
            output_prediction = np.reshape(output_prediction, original_shape)

        return output_prediction

    """
    Returns np array of predicted heatmaps for a given image and model

    ## Parameters

    X_batch : {list of ndarrays}
        A list of images which were used as input to the model

    predict_using_flip : {bool}
        Perform prediction using a flipped version of the input. NOTE the output will be transformed
        back into the original image coordinate space. Treat this output as you would a normal prediction.

    ## Returns:
    output shape is (num_hg_blocks, X_batch_size, 64, 64, 17)
    """
    def predict_heatmaps(self, X_batch, predict_using_flip=False):
        def _predict(X_batch):
            # Instead of calling model.predict or model.predict_on_batch, we call model by itself.
            # See https://stackoverflow.com/questions/66271988/warningtensorflow11-out-of-the-last-11-calls-to-triggered-tf-function-retracin
            # This should fix our memory leak in keras
            return np.array(self.model.predict_on_batch(X_batch))

        # X_batch has dimensions (batch, x, y, channels)
        # Run both original and flipped image through and average the predictions
        # Typically increases accuracy by a few percent
        if predict_using_flip:
            # Horizontal flip each image in batch
            X_batch_flipped = X_batch[:,:,::-1,:]

            # Feed flipped image into model
            # output shape is (num_hg_blocks, X_batch_size, 64, 64, 17)
            predicted_heatmaps_batch_flipped = _predict(X_batch_flipped)

            # indices to flip order of Left and Right heatmaps [0, 2, 1, 4, 3, 6, 5, 8, 7, etc]
            reverse_LR_indices = [0] + [2*x-y for x in range(1,9) for y in range(2)]

            # reverse horizontal flip AND reverse left/right heatmaps
            predicted_heatmaps_batch = predicted_heatmaps_batch_flipped[:,:,:,::-1,reverse_LR_indices]
        else:
            predicted_heatmaps_batch = _predict(X_batch)

        return predicted_heatmaps_batch

    """
    Visualize the set of keypoints on the model image.

    Note, it is assumed that the images have the same dimension domain as the keypoints.
    (i.e., they keypoint (x,y) should point to the corresponding pixel on the image.)

    ## Parameters

    X_batch : {list of ndarrays}
        A list of images, with the same dimensionality as the keypoints. This means
        if the keypoints are relative to a (256 x 256) image, each element of X_batch must be the same
        dimension.

    keypoints_batch : {list of lists}
        Each element consists of a list of keypoints, with each keypoint having the components of (x,y,score).

    img_id_batch : {list of strings}
        A list of image names. These should not contain the extension, epoch, or type. (Purely image ID)

    show_skeleton : {bool}
        If true, connects joints together (if possible) to construct a COCO-format skeleton
    """
    def visualize_keypoints(self, X_batch, keypoints_batch, show_skeleton=True):

        for i in range(len(X_batch)):
            X = X_batch[i]
            keypoints = keypoints_batch[i]

            # Plot predicted keypoints on bounding box image
            x_left = []
            y_left = []
            x_right = []
            y_right = []
            valid = np.zeros(NUM_COCO_KEYPOINTS)

            for i in range(NUM_COCO_KEYPOINTS):
                if keypoints[i,0] != 0 and keypoints[i,1] != 0:
                    valid[i] = 1

                    if i % 2 == 0:
                        x_right.append(keypoints[i,0])
                        y_right.append(keypoints[i,1])
                    else:
                        x_left.append(keypoints[i,0])
                        y_left.append(keypoints[i,1])
            with _lock:
                if show_skeleton:
                    color_index = 0
                    for i in range(len(COCO_SKELETON)):
                        # joint a to joint b
                        a = COCO_SKELETON[i, 0]
                        b = COCO_SKELETON[i, 1]

                        # if both are valid keypoints
                        if valid[a] and valid[b]:
                            # linewidth = 5, linestyle = "--",
                            plt.plot([keypoints[a,0],keypoints[b,0]], [keypoints[a,1], keypoints[b,1]], color = COLOUR_MAP[color_index % 10])

                            color_index += 1

                plt.scatter(x_left,y_left, color=COLOUR_MAP[0])
                plt.scatter(x_right,y_right, color=COLOUR_MAP[4])
                # https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
                plt.axis('off')
                plt.imshow(X)

                plt.tight_layout(w_pad=0)

                # https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False, dpi=300)
                plt.close('all')

            # Ensure Image reads from the beginning
            buf.seek(0)
            im = Image.open(buf).convert('RGB')

            # Turn heatmap into numpy array.
            # NOTE that the read image size is too large because of plt's default size.
            visualized = np.array(im)
            buf.close()
            im.close()
            return visualized

    def heatmaps_to_keypoints_batch(self, heatmaps_batch, threshold=HM_TO_KP_THRESHOLD):
        keypoints_batch = []

        # dimensions are (num_hg_blocks, batch, x, y, keypoint)
        for i in range(heatmaps_batch.shape[1]):
            # Get predicted keypoints from last hourglass (last element of list)
            # Dimensions are (hourglass_layer, batch, x, y, keypoint)
            keypoints = self.heatmaps_to_keypoints(heatmaps_batch[-1, i, :, :, :])

            keypoints_batch.append(keypoints)

        return np.array(keypoints_batch)

    # Resources for heatmaps to keypoints
    # https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/blob/eddf0ae15715a88d7859847cfff5f5092b260ae1/src/eval/heatmap_process.py#L5
    # https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/56707252501c73b2bf2aac8fff3e22760fd47dca/hourglass/postprocess.py#L17

    ### Returns np array of predicted keypoints from one image's heatmaps
    def heatmaps_to_keypoints(self, heatmaps, threshold=HM_TO_KP_THRESHOLD):
        keypoints = np.zeros((NUM_COCO_KEYPOINTS, NUM_COCO_KP_ATTRBS))
        for i in range(NUM_COCO_KEYPOINTS):
            hmap = heatmaps[:,:,i]
            # Resize heatmap from Output DIM to Input DIM
            resized_hmap = cv2.resize(hmap, INPUT_DIM, interpolation = cv2.INTER_LINEAR)
            # Do a heatmap blur with gaussian_filter
            resized_hmap = gaussian_filter(resized_hmap, REVERSE_HEATMAP_SIGMA)

            # Get peak point (brightest area) in heatmap with 3x3 max filter
            peaks = self._non_max_supression(resized_hmap, threshold, windowSize=3)

            # Choose the max point in heatmap (we only pick 1 keypoint in each heatmap)
            # and get its coordinates and confidence
            y, x = np.unravel_index(np.argmax(peaks), peaks.shape)

            # reduce threshold since non-maximum suppression may have reduced the maximum value
            # values below this threshold have already been suppressed to zero so this shouldnt
            # affect the conversion of heatmap to keypoint
            if peaks[y, x] > HM_TO_KP_THRESHOLD_POST_FILTER:
                conf = peaks[y, x]
            else:
                x, y, conf = 0, 0, 0

            keypoints[i, 0] = x
            keypoints[i, 1] = y
            keypoints[i, 2] = conf

        return keypoints

    def _non_max_supression(self, plain, threshold, windowSize=3):
        # Clear values less than threshold
        under_thresh_indices = plain < threshold
        plain[under_thresh_indices] = 0
        return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))

"""
Runs the model for any general file. This aims to extend the DataGenerator output format for arbitrary images

## Parameters:
img_path : {string-typed} path to image
    Note this image must be square, and centered around the person you wish to retrieve predictions for.

num_hg_blocks : {int}
    number of hourglass blocks to generate dummy ground truth data

bbox : {tuple of element type int or float}
    optional bounding box info, anchored at top left of image, of elements (x,y,w,h)
"""
def load_and_preprocess_img(img_path, num_hg_blocks, bbox=None):
    img = Image.open(img_path).convert('RGB')

    # Required because PIL will read EXIF tags about rotation by default. We want to
    # preserve the input image rotation so we manually apply the rotation if required.
    # See https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image/
    # and the answer I used: https://stackoverflow.com/a/63798032
    img = ImageOps.exif_transpose(img)

    if bbox is None:
        w, h = img.size

        if w != h:
            # if the image is not square
            # Indexed so upper left corner is (0,0)
            bbox = data_generator.transform_bbox_square((0, 0, w, h))

    if bbox is not None:
        # If a bounding box is provided, use it
        bbox = np.array(bbox, dtype=int)

        # Crop with box of order left, upper, right, lower
        img = img.crop(box=bbox)

    new_img = cv2.resize(np.array(img), INPUT_DIM,
                        interpolation=cv2.INTER_LINEAR)

    # Add a 'batch' axis
    X_batch = np.expand_dims(new_img.astype('float'), axis=0)

    # Add dummy heatmap "ground truth", duplicated 'num_hg_blocks' times
    y_batch = [np.zeros((1, *(OUTPUT_DIM), NUM_COCO_KEYPOINTS), dtype='float') for _ in range(num_hg_blocks)]

    # Normalize input image
    X_batch /= 255
    return X_batch, y_batch

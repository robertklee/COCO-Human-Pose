import csv
import io
import json
import os
import re
from functools import lru_cache

import cv2
import matplotlib
import matplotlib.pyplot as plt
import threading
_lock = threading.Lock()
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from tensorflow import keras
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, maximum_filter

import data_generator
import evaluation
import util
from constants import *


# TODO: Refactor this class to use the existing (COCO API dependent) evaluation_wrapper, evaluation, and hourglass code

class AppHelper():
    def __init__(self, model_weights, model_json) -> None:
        self._load_model(model_json=model_json, model_weights=model_weights)

    def predict_in_memory(self, img, visualize_scatter=True, visualize_skeleton=True, average_flip_prediction=True):
        # Number of hg blocks doesn't matter
        X_batch, y_stacked, _crop_info = load_and_preprocess_img(img, 1)
        y_batch = y_stacked[0] # take first hourglass section
        img_id_batch = None

        return self._predict_and_visualize(
            X_batch,
            visualize_scatter=visualize_scatter,
            visualize_skeleton=visualize_skeleton,
            average_flip_prediction=average_flip_prediction
        )

    def predict_in_memory_fullres(self, img_path, average_flip_prediction=True):
        """Predict keypoints and return them mapped to original image coordinates.

        Returns (orig_batch, fullres_keypoints_batch) for use with visualize_keypoints().
        """
        X_batch, _y_stacked, crop_info = load_and_preprocess_img(img_path, 1)

        predicted_heatmaps_batch = self.predict_heatmaps(X_batch)
        keypoints_batch = self.heatmaps_to_keypoints_batch(predicted_heatmaps_batch)

        if average_flip_prediction:
            predicted_heatmaps_batch_2 = self.predict_heatmaps(X_batch=X_batch, predict_using_flip=True)
            keypoints_batch_2 = self.heatmaps_to_keypoints_batch(predicted_heatmaps_batch_2)
            for i in range(keypoints_batch.shape[0]):
                keypoints_batch[i] = self._average_LR_flip_predictions(keypoints_batch[i], keypoints_batch_2[i], coco_format=False)

        # Map keypoints from 256x256 model space back to original image coordinates
        bbox = crop_info['bbox']  # (left, upper, right, lower)
        crop_w = crop_info['crop_w']
        crop_h = crop_info['crop_h']
        scale_x = crop_w / INPUT_DIM[0]
        scale_y = crop_h / INPUT_DIM[1]
        anchor_x = bbox[0]
        anchor_y = bbox[1]

        fullres_keypoints_batch = keypoints_batch.copy()
        for i in range(fullres_keypoints_batch.shape[0]):
            for j in range(NUM_COCO_KEYPOINTS):
                kp_x = fullres_keypoints_batch[i, j, 0]
                kp_y = fullres_keypoints_batch[i, j, 1]
                if kp_x != 0 or kp_y != 0:
                    fullres_keypoints_batch[i, j, 0] = kp_x * scale_x + anchor_x
                    fullres_keypoints_batch[i, j, 1] = kp_y * scale_y + anchor_y

        # Load original full-res image for overlay
        with Image.open(img_path) as orig_img:
            orig_img = ImageOps.exif_transpose(orig_img.convert('RGB'))
            orig_array = np.array(orig_img) / 255.0

        orig_batch = np.expand_dims(orig_array, axis=0)
        return orig_batch, fullres_keypoints_batch

    def _load_model(self, model_json, model_weights):
        from hourglass_blocks import create_hourglass_network, bottleneck_block
        from tensorflow.keras.models import model_from_json

        with open(model_json) as f:
            model_config = json.load(f)

        is_keras2 = model_weights.endswith('.hdf5')

        if is_keras2:
            # Keras 2 JSON cannot be deserialized by Keras 3's model_from_json.
            # Rebuild architecture from code and load only the weights.
            num_stacks = int(re.search(r'stacks_(\d+)', model_json).group(1))

            activation = DEFAULT_ACTIVATION
            for layer in model_config['config']['layers']:
                if '_conv_1x1_parts' in layer.get('name', ''):
                    activation = layer['config']['activation']
                    break

            self.model = create_hourglass_network(
                NUM_COCO_KEYPOINTS, num_stacks, NUM_CHANNELS,
                INPUT_DIM, OUTPUT_DIM, bottleneck_block, activation
            )
            self.model.load_weights(model_weights)
        else:
            # Keras 3 JSON can be loaded directly
            self.model = model_from_json(json.dumps(model_config))
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

        for idx in range(len(X_batch)):
            X = X_batch[idx]
            keypoints = keypoints_batch[idx]

            # Convert float [0,1] image to uint8 for OpenCV drawing
            canvas = (np.clip(X, 0, 1) * 255).astype(np.uint8).copy()
            h, w = canvas.shape[:2]

            # Scale drawing sizes relative to image dimensions
            base_dim = max(h, w)
            line_thickness = max(2, int(base_dim / 120))
            kp_radius = max(3, int(base_dim / 80))
            kp_border = max(1, kp_radius // 3)

            valid = np.zeros(NUM_COCO_KEYPOINTS)
            for j in range(NUM_COCO_KEYPOINTS):
                if keypoints[j, 0] != 0 and keypoints[j, 1] != 0:
                    valid[j] = 1

            # Draw skeleton bones (below keypoint dots)
            if show_skeleton:
                for j in range(len(COCO_SKELETON)):
                    a = COCO_SKELETON[j, 0]
                    b = COCO_SKELETON[j, 1]
                    if valid[a] and valid[b]:
                        pt1 = (int(keypoints[a, 0]), int(keypoints[a, 1]))
                        pt2 = (int(keypoints[b, 0]), int(keypoints[b, 1]))
                        cv2.line(canvas, pt1, pt2, SKELETON_BONE_COLORS[j],
                                 line_thickness, lineType=cv2.LINE_AA)

            # Draw keypoint dots on top of skeleton
            for j in range(NUM_COCO_KEYPOINTS):
                if valid[j]:
                    pt = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                    # Dark border for visibility against any background
                    cv2.circle(canvas, pt, kp_radius, (0, 0, 0), -1,
                               lineType=cv2.LINE_AA)
                    # Colored fill per keypoint
                    cv2.circle(canvas, pt, kp_radius - kp_border,
                               KEYPOINT_COLORS[j], -1, lineType=cv2.LINE_AA)

            return canvas

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
            peaks = self._non_max_supression(resized_hmap.copy(), threshold, windowSize=3)

            # Find the peak location for confidence check
            peak_y, peak_x = np.unravel_index(np.argmax(peaks), peaks.shape)

            if peaks[peak_y, peak_x] > HM_TO_KP_THRESHOLD_POST_FILTER:
                conf = peaks[peak_y, peak_x]
                # Use weighted centroid on the smoothed heatmap (pre-NMS) for sub-pixel accuracy
                x, y = self._weighted_centroid(resized_hmap, peak_y, peak_x)
            else:
                x, y, conf = 0, 0, 0

            keypoints[i, 0] = x
            keypoints[i, 1] = y
            keypoints[i, 2] = conf

        return keypoints

    def _weighted_centroid(self, heatmap, peak_y, peak_x,
                           brightness_k=WEIGHTED_CENTROID_BRIGHTNESS_K,
                           spatial_k=WEIGHTED_CENTROID_SPATIAL_K):
        """Compute the weighted centroid of the heatmap region near the peak.

        The region is defined by pixels that are both:
        - Within brightness_k std devs of the peak brightness value
        - Within spatial_k std devs of distance from the peak
        """
        peak_val = heatmap[peak_y, peak_x]

        # Brightness constraint: include pixels within k std devs of the peak
        nonzero_vals = heatmap[heatmap > 0]
        if len(nonzero_vals) < 2:
            return peak_x, peak_y
        brightness_std = np.std(nonzero_vals)
        brightness_threshold = peak_val - brightness_k * brightness_std
        brightness_mask = heatmap >= brightness_threshold

        # Spatial constraint: include pixels within k std devs of distance from peak
        ys, xs = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
        distances = np.sqrt((ys - peak_y) ** 2 + (xs - peak_x) ** 2)
        bright_distances = distances[brightness_mask]
        if len(bright_distances) < 2:
            return peak_x, peak_y
        spatial_std = np.std(bright_distances)
        if spatial_std == 0:
            return peak_x, peak_y
        spatial_mask = distances <= spatial_k * spatial_std

        # Combined region
        region_mask = brightness_mask & spatial_mask
        weights = heatmap[region_mask]

        if weights.sum() == 0:
            return peak_x, peak_y

        region_ys = ys[region_mask]
        region_xs = xs[region_mask]
        centroid_x = np.average(region_xs, weights=weights)
        centroid_y = np.average(region_ys, weights=weights)

        return centroid_x, centroid_y

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
    with Image.open(img_path) as img:
        img = img.convert('RGB')

        # Required because PIL will read EXIF tags about rotation by default. We want to
        # preserve the input image rotation so we manually apply the rotation if required.
        # See https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image/
        # and the answer I used: https://stackoverflow.com/a/63798032
        img = ImageOps.exif_transpose(img)

        orig_w, orig_h = img.size

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
        else:
            bbox = np.array([0, 0, orig_w, orig_h], dtype=int)

        crop_w, crop_h = img.size

        new_img = cv2.resize(np.array(img), INPUT_DIM,
                            interpolation=cv2.INTER_LINEAR)

    # Add a 'batch' axis
    X_batch = np.expand_dims(new_img.astype('float'), axis=0)

    # Add dummy heatmap "ground truth", duplicated 'num_hg_blocks' times
    y_batch = [np.zeros((1, *(OUTPUT_DIM), NUM_COCO_KEYPOINTS), dtype='float') for _ in range(num_hg_blocks)]

    # Normalize input image
    X_batch /= 255

    # Crop metadata for mapping keypoints back to original image coordinates
    crop_info = {
        'bbox': bbox,           # (left, upper, right, lower) used for cropping
        'crop_w': crop_w,       # width of cropped region
        'crop_h': crop_h,       # height of cropped region
        'orig_w': orig_w,
        'orig_h': orig_h,
    }

    return X_batch, y_batch, crop_info

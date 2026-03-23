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
from tensorflow import keras
from PIL import Image, ImageOps

import data_generator
import evaluation
import util
from constants import *

# Resolution caps for each visualization output, avoiding excessive memory on
# high-resolution uploads while keeping the skeleton as close to original as possible.
MAX_PIXELS_SKELETON = 12_000_000  # 12 MP (~36 MB uint8 RGB)
MAX_DIM_SCATTER = 2560
MAX_DIM_HEATMAP = 1280

# Reject images with extreme aspect ratios. The model input is 256×256; non-square
# images are letterboxed into a square crop, so extreme ratios waste most of the input.
# Human bounding boxes are roughly upright, so even 3:2 is generous.
MAX_ASPECT_RATIO = 1.5  # 6:4 (i.e. 3:2)


def downscale_for_display(orig_batch, keypoints_batch, crop_info, max_dim):
    """Return copies of orig_batch / keypoints / crop_info downscaled to *max_dim*.

    If the image is already within *max_dim*, the inputs are returned unchanged
    (no copy is made).  Model inference is never affected — only the display
    arrays are resized.
    """
    h, w = orig_batch.shape[1], orig_batch.shape[2]
    if max(w, h) <= max_dim:
        return orig_batch, keypoints_batch, crop_info

    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize each image in the batch
    resized = np.stack([
        cv2.resize(orig_batch[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
        for i in range(orig_batch.shape[0])
    ])

    kp = keypoints_batch.copy()
    kp[:, :, 0] *= scale
    kp[:, :, 1] *= scale

    ci = dict(crop_info)
    ci['bbox'] = (crop_info['bbox'].astype(np.float64) * scale).astype(int)
    ci['crop_w'] = int(crop_info['crop_w'] * scale)
    ci['crop_h'] = int(crop_info['crop_h'] * scale)

    return resized, kp, ci


# TODO: Refactor this class to use the existing (COCO API dependent) evaluation_wrapper, evaluation, and hourglass code

class AppHelper():
    def __init__(self, model_weights, model_json) -> None:
        self._load_model(model_json=model_json, model_weights=model_weights)

    def predict_in_memory(self, img, visualize_scatter=True, visualize_skeleton=True, average_flip_prediction=True):
        # Number of hg blocks doesn't matter
        X_batch, y_stacked, _crop_info = util.load_and_preprocess_img(img, 1)
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

        Returns (orig_batch, fullres_keypoints_batch, heatmaps, crop_info) where
        heatmaps is the last hourglass stack output with shape (batch, 64, 64, 17)
        and crop_info contains the bbox and crop dimensions used for preprocessing.
        """
        X_batch, _y_stacked, crop_info = util.load_and_preprocess_img(img_path, 1)

        aspect_ratio = max(crop_info['orig_w'], crop_info['orig_h']) / \
                        max(1, min(crop_info['orig_w'], crop_info['orig_h']))
        if aspect_ratio > MAX_ASPECT_RATIO:
            raise ValueError(
                f"Image aspect ratio {aspect_ratio:.1f}:1 is too extreme "
                f"(max {MAX_ASPECT_RATIO:.0f}:1). Panoramic or very narrow "
                f"images produce unreliable pose estimates because the model "
                f"input is square. Please crop the image closer to the subject."
            )

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

        # Load original image for overlay, capped to MAX_PIXELS_SKELETON to bound
        # memory for extreme resolutions (e.g. 100 MP).
        with Image.open(img_path) as orig_img:
            orig_img = ImageOps.exif_transpose(orig_img.convert('RGB'))
            w, h = orig_img.size
            if w * h > MAX_PIXELS_SKELETON:
                display_scale = (MAX_PIXELS_SKELETON / (w * h)) ** 0.5
                new_w, new_h = int(w * display_scale), int(h * display_scale)
                orig_img = orig_img.resize((new_w, new_h), Image.LANCZOS)
                fullres_keypoints_batch[:, :, 0] *= display_scale
                fullres_keypoints_batch[:, :, 1] *= display_scale
                crop_info['bbox'] = (crop_info['bbox'].astype(np.float64) * display_scale).astype(int)
                crop_info['crop_w'] = int(crop_info['crop_w'] * display_scale)
                crop_info['crop_h'] = int(crop_info['crop_h'] * display_scale)
            orig_array = np.array(orig_img, dtype=np.uint8)

        orig_batch = np.expand_dims(orig_array, axis=0)
        # Last hourglass stack heatmaps: shape (batch, 64, 64, 17)
        last_heatmaps = predicted_heatmaps_batch[-1]
        return orig_batch, fullres_keypoints_batch, last_heatmaps, crop_info

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
        return util.average_LR_flip_predictions(prediction_1, prediction_2, coco_format)

    def predict_heatmaps(self, X_batch, predict_using_flip=False):
        return util.predict_heatmaps(self.model, X_batch, predict_using_flip)

    def visualize_keypoints(self, X_batch, keypoints_batch, show_skeleton=True):
        """Draw keypoints (and optionally skeleton) on the first image in the batch.

        Only the first image is processed; callers always pass batch-size-1
        arrays from single-image inference.
        """
        X = X_batch[0]
        keypoints = keypoints_batch[0]

        # Accept both uint8 [0,255] and float [0,1] images
        if X.dtype == np.uint8:
            canvas = X.copy()
        else:
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

    def visualize_heatmap(self, image, heatmap, joint_index, crop_info=None,
                          alpha=0.7, image_brightness=0.4):
        """Overlay a single joint's heatmap on the image.

        Parameters
        ----------
        image : ndarray
            Original image as uint8 [0,255] or float [0,1] with shape (H, W, 3).
        heatmap : ndarray
            Single-joint heatmap with shape (64, 64).
        joint_index : int
            Index into COCO_KEYPOINT_LABEL_ARR (used for the keypoint color).
        crop_info : dict or None
            Crop metadata from predict_in_memory_fullres. When provided, the
            heatmap is placed only within the crop region on the full image.
        alpha : float
            Blend factor for the heatmap overlay.
        image_brightness : float
            Dim the base image (0 = black, 1 = original brightness).

        Returns
        -------
        ndarray : uint8 RGB image with the heatmap overlay.
        """
        h, w = image.shape[:2]
        if image.dtype == np.uint8:
            base_img = (image * image_brightness).astype(np.uint8)
        else:
            base_img = (np.clip(image, 0, 1) * 255 * image_brightness).astype(np.uint8)

        if crop_info is not None:
            # Heatmap is relative to the crop region, not the full image.
            # Resize heatmap to the crop dimensions, then place it on a
            # full-image-sized canvas at the correct position.
            bbox = crop_info['bbox']  # (left, upper, right, lower)
            crop_w = crop_info['crop_w']
            crop_h = crop_info['crop_h']

            hm_crop = cv2.resize(heatmap, (crop_w, crop_h),
                                 interpolation=cv2.INTER_LINEAR)
            hm_full = np.zeros((h, w), dtype=np.float32)

            # Clamp placement to image bounds
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)
            src_x = x1 - int(bbox[0])
            src_y = y1 - int(bbox[1])
            hm_full[y1:y2, x1:x2] = hm_crop[src_y:src_y + (y2 - y1),
                                              src_x:src_x + (x2 - x1)]
            hm_resized = hm_full
        else:
            hm_resized = cv2.resize(heatmap, (w, h),
                                    interpolation=cv2.INTER_LINEAR)

        # Scale by absolute confidence rather than self-normalizing.
        # This way weak predictions produce dim heatmaps and strong ones are bright.
        peak_val = hm_resized.max()
        confidence = np.clip(peak_val, 0, 1)
        if confidence < 1e-4:
            # No meaningful activation — return the dimmed base image
            return base_img

        hm_norm = np.clip(hm_resized / (peak_val + 1e-8), 0, 1)

        # Gamma correction to boost mid-level activations
        hm_norm = np.power(hm_norm, 0.6)

        # Scale the entire overlay by the peak confidence
        hm_norm = hm_norm * confidence

        hm_uint8 = (hm_norm * 255).astype(np.uint8)

        # Apply a colormap (hot: black → red → yellow → white)
        hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)
        hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)

        # Blend proportionally to heatmap activation
        mask = hm_norm[..., np.newaxis]
        blended = (base_img * (1 - alpha * mask) + hm_colored * alpha * mask).astype(np.uint8)

        return blended

    def heatmaps_to_keypoints_batch(self, heatmaps_batch, threshold=HM_TO_KP_THRESHOLD):
        return util.heatmaps_to_keypoints_batch(heatmaps_batch, threshold)

    def heatmaps_to_keypoints(self, heatmaps, threshold=HM_TO_KP_THRESHOLD):
        return util.heatmaps_to_keypoints(heatmaps, threshold)

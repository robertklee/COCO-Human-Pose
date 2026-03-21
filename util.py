import os
import re
import sys
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, maximum_filter

import data_generator
from constants import *

MODEL_ARCHITECTURE_JSON_REGEX = f'^{HPE_HOURGLASS_STACKS_PREFIX}.*\\.json$'
MODEL_CHECKPOINT_REGEX = f'^{HPE_EPOCH_PREFIX}([\\d]+).*\\.keras$'

def str_to_enum(EnumClass, str):
    try:
        enum_ = EnumClass[str]
    except KeyError:
        return None
    return enum_

def validate_enum(EnumClass, str):
    enum_ = str_to_enum(EnumClass=EnumClass, str=str)

    if enum_ is None:
        print(f'\'{str}\' was not found in possible options for Enum class: {EnumClass.__name__}.')
        print('Available options are:')
        options = [name for name, _ in EnumClass.__members__.items()]
        print(options)
        exit(1)
    return True

def is_highest_epoch_file(model_base_dir, model_subdir, epoch_):
    highest_epoch = get_highest_epoch_file(model_base_dir, model_subdir)

    return epoch_ >= highest_epoch

def get_highest_epoch_file(model_base_dir, model_subdir):
    enclosing_dir = os.path.join(model_base_dir, model_subdir)

    files = os.listdir(enclosing_dir)

    highest_epoch = -1

    for f in files:
        match = re.match(MODEL_CHECKPOINT_REGEX, f)

        if match:
            epoch = int(match.group(1))

            if epoch > highest_epoch:
                highest_epoch = epoch

    return highest_epoch

def get_all_epochs(model_base_dir, model_subdir):
    enclosing_dir = os.path.join(model_base_dir, model_subdir)

    files = os.listdir(enclosing_dir)

    model_saved_weights = {}
    for f in files:
        match = re.match(MODEL_CHECKPOINT_REGEX, f)

        if match:
            epoch = int(match.group(1))
            model_saved_weights[epoch] = f

    return model_saved_weights

def find_resume_json_weights_str(model_base_dir, model_subdir, resume_epoch):
    enclosing_dir = os.path.join(model_base_dir, model_subdir)

    files = os.listdir(enclosing_dir)

    model_jsons         = [f for f in files if re.match(MODEL_ARCHITECTURE_JSON_REGEX, f)]
    model_saved_weights = {}
    for f in files:
        match = re.match(MODEL_CHECKPOINT_REGEX, f)

        if match:
            epoch = int(match.group(1))
            model_saved_weights[epoch] = f

    assert len(model_jsons) > 0, "Subdirectory does not contain any model architecture json files"
    assert len(model_saved_weights) > 0, "Subdirectory does not contain any saved model weights"

    if resume_epoch is None or resume_epoch <= 0:
        resume_epoch = max(k for k, _ in model_saved_weights.items())

        print(f'No epoch number provided. Automatically using largest epoch number {resume_epoch:3d}.')

    resume_json = os.path.join(enclosing_dir, model_jsons[0])
    resume_weights = os.path.join(enclosing_dir, model_saved_weights[resume_epoch])

    print('Found model json:                  {}\n'.format(resume_json))
    print('Found model weights for epoch {epoch:3d}: {weight_file_name}\n'.format(epoch=resume_epoch, weight_file_name=resume_weights))

    assert os.path.exists(resume_json)
    assert os.path.exists(resume_weights)

    return resume_json, resume_weights, resume_epoch

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

# https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
def print_progress_bar(percent, label=None):
    # If a label is provided
    if label is not None and label != '':
        label = label + ': '
    else:
        label = ''

    width = 20 # This width is fixed.
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("%s[%-20s] %d%%" % (label, '='*int(percent*width), int(percent*100)))
    sys.stdout.flush()


def non_max_suppression(plain, threshold, windowSize=3):
    """Zero out values below threshold, then keep only local maxima."""
    under_thresh_indices = plain < threshold
    plain[under_thresh_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))


def weighted_centroid(heatmap, peak_y, peak_x,
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


# Resources for heatmaps to keypoints
# https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/blob/eddf0ae15715a88d7859847cfff5f5092b260ae1/src/eval/heatmap_process.py#L5
# https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/56707252501c73b2bf2aac8fff3e22760fd47dca/hourglass/postprocess.py#L17

def heatmaps_to_keypoints(heatmaps, threshold=HM_TO_KP_THRESHOLD):
    """Return np array of predicted keypoints from one image's heatmaps."""
    keypoints = np.zeros((NUM_COCO_KEYPOINTS, NUM_COCO_KP_ATTRBS))
    for i in range(NUM_COCO_KEYPOINTS):
        hmap = heatmaps[:,:,i]
        # Resize heatmap from Output DIM to Input DIM
        resized_hmap = cv2.resize(hmap, INPUT_DIM, interpolation = cv2.INTER_LINEAR)
        # Do a heatmap blur with gaussian_filter
        resized_hmap = gaussian_filter(resized_hmap, REVERSE_HEATMAP_SIGMA)

        # Get peak point (brightest area) in heatmap with 3x3 max filter
        peaks = non_max_suppression(resized_hmap.copy(), threshold, windowSize=3)

        # Find the peak location for confidence check
        peak_y, peak_x = np.unravel_index(np.argmax(peaks), peaks.shape)

        if peaks[peak_y, peak_x] > HM_TO_KP_THRESHOLD_POST_FILTER:
            conf = peaks[peak_y, peak_x]
            # Use weighted centroid on the smoothed heatmap (pre-NMS) for sub-pixel accuracy
            x, y = weighted_centroid(resized_hmap, peak_y, peak_x)
        else:
            x, y, conf = 0, 0, 0

        keypoints[i, 0] = x
        keypoints[i, 1] = y
        keypoints[i, 2] = conf

    return keypoints


def heatmaps_to_keypoints_batch(heatmaps_batch, threshold=HM_TO_KP_THRESHOLD):
    """Return np array of predicted keypoints for a batch of heatmaps.

    Parameters
    ----------
    heatmaps_batch : ndarray
        Shape (num_hg_blocks, batch, x, y, keypoint).

    Returns
    -------
    ndarray of shape (batch, NUM_COCO_KEYPOINTS, NUM_COCO_KP_ATTRBS)
    """
    keypoints_batch = []

    # dimensions are (num_hg_blocks, batch, x, y, keypoint)
    for i in range(heatmaps_batch.shape[1]):
        # Get predicted keypoints from last hourglass (last element of list)
        keypoints = heatmaps_to_keypoints(heatmaps_batch[-1, i, :, :, :])
        keypoints_batch.append(keypoints)

    return np.array(keypoints_batch)


def predict_heatmaps(model, X_batch, predict_using_flip=False):
    """Return predicted heatmaps for a batch of images.

    Parameters
    ----------
    model : keras.Model
        The loaded hourglass model.
    X_batch : ndarray
        Input images with shape (batch, x, y, channels).
    predict_using_flip : bool
        When True, horizontally flip the input, predict, then undo the flip
        and swap left/right keypoint channels so the output is in the original
        coordinate space.

    Returns
    -------
    ndarray of shape (num_hg_blocks, batch, 64, 64, 17)
    """
    def _predict(X_batch):
        # predict_on_batch avoids the memory-leak caused by repeated
        # model.predict / model.__call__ retracing.
        # See https://stackoverflow.com/questions/66271988
        return np.array(model.predict_on_batch(X_batch))

    if predict_using_flip:
        X_batch_flipped = X_batch[:,:,::-1,:]

        predicted_heatmaps_batch_flipped = _predict(X_batch_flipped)

        # indices to flip order of Left and Right heatmaps [0, 2, 1, 4, 3, 6, 5, 8, 7, …]
        reverse_LR_indices = [0] + [2*x-y for x in range(1,9) for y in range(2)]

        # reverse horizontal flip AND reverse left/right heatmaps
        predicted_heatmaps_batch = predicted_heatmaps_batch_flipped[:,:,:,::-1,reverse_LR_indices]
    else:
        predicted_heatmaps_batch = _predict(X_batch)

    return predicted_heatmaps_batch


def average_LR_flip_predictions(prediction_1, prediction_2, coco_format=True):
    """Average keypoint predictions from an original and horizontally-flipped input.

    Skips keypoints where either prediction has confidence below
    HM_TO_KP_THRESHOLD to avoid averaging valid detections with (0,0,0).

    Parameters
    ----------
    prediction_1, prediction_2 : ndarray
        Keypoint arrays — either shaped (NUM_COCO_KEYPOINTS, 3) or flat COCO
        format (NUM_COCO_KEYPOINTS * 3,).
    coco_format : bool
        When True, returns a flat array with visibility set to 1 for detected
        keypoints (evaluation consumer).  When False, returns an array reshaped
        to the original input shape with raw averaged confidence (demo app
        consumer).
    """
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

    if not coco_format:
        output_prediction = np.reshape(output_prediction, original_shape)

    return output_prediction


def load_and_preprocess_img(img_path, num_hg_blocks, bbox=None):
    """Load an image, crop/resize to model input dimensions, and return batch arrays.

    Used by both the inference demo app (which needs crop_info to map
    predictions back to full-resolution coordinates) and the evaluation
    pipeline (which can ignore crop_info).

    Parameters
    ----------
    img_path : str
        Path to the image file.
    num_hg_blocks : int
        Number of hourglass blocks — used to generate dummy ground truth.
    bbox : tuple or None
        Optional bounding box (x, y, w, h) anchored at top-left.

    Returns
    -------
    X_batch : ndarray
        Preprocessed image with shape (1, *INPUT_DIM, 3), normalised to [0,1].
    y_batch : list of ndarray
        Dummy ground truth heatmaps, one per hourglass block.
    crop_info : dict
        Crop metadata: bbox, crop_w, crop_h, orig_w, orig_h.
    """
    with Image.open(img_path) as img:
        img = img.convert('RGB')

        # Preserve original rotation from EXIF metadata.
        # See https://stackoverflow.com/a/63798032
        img = ImageOps.exif_transpose(img)

        orig_w, orig_h = img.size

        if bbox is None:
            w, h = img.size

            if w != h:
                # if the image is not square
                bbox = data_generator.transform_bbox_square((0, 0, w, h))

        if bbox is not None:
            bbox = np.array(bbox, dtype=int)
            img = img.crop(box=bbox)
        else:
            bbox = np.array([0, 0, orig_w, orig_h], dtype=int)

        crop_w, crop_h = img.size

        new_img = cv2.resize(np.array(img), INPUT_DIM,
                            interpolation=cv2.INTER_LINEAR)

    X_batch = np.expand_dims(new_img.astype('float'), axis=0)

    y_batch = [np.zeros((1, *(OUTPUT_DIM), NUM_COCO_KEYPOINTS), dtype='float') for _ in range(num_hg_blocks)]

    X_batch /= 255

    crop_info = {
        'bbox': bbox,
        'crop_w': crop_w,
        'crop_h': crop_h,
        'orig_w': orig_w,
        'orig_h': orig_h,
    }

    return X_batch, y_batch, crop_info


if __name__ == "__main__":
    from time import sleep

    for i in range(21):
        print_progress_bar(i/20.0, label="test")
        sleep(0.25)
    print()

    for i in range(21):
        print_progress_bar(i/20.0)
        sleep(0.25)
    print()

    for i in range(21):
        print_progress_bar(i/20.0, label='')
        sleep(0.25)
    print()

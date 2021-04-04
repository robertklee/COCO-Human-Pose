import json
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from pycocotools.cocoeval import COCOeval
from scipy.ndimage import gaussian_filter, maximum_filter

import data_generator
import HeatMap  # https://github.com/LinShanify/HeatMap
import hourglass
import util
from constants import *


class Evaluation():

    def __init__(self, model_sub_dir, epoch, model_base_dir=DEFAULT_MODEL_BASE_DIR, output_base_dir=DEFAULT_OUTPUT_BASE_DIR):
        # automatically retrieve json and weights
        self.model_sub_dir=model_sub_dir
        self.epoch=epoch
        match = re.match(r'(.*)(_resume_.*$)', model_sub_dir)
        if match:
            self.output_sub_dir = os.path.join(output_base_dir, match.group(1), str(self.epoch))
        else:
            self.output_sub_dir = os.path.join(output_base_dir, self.model_sub_dir, str(self.epoch))
        if not os.path.exists(self.output_sub_dir):
            os.makedirs(self.output_sub_dir)

        self.model_json, self.weights, _ = util.find_resume_json_weights_str(model_base_dir, model_sub_dir, epoch)
        self.num_hg_blocks = int(re.match(r'.*stacks_([\d]+)_.*',self.model_json).group(1))
        h = hourglass.HourglassNet(NUM_COCO_KEYPOINTS,self.num_hg_blocks,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
        h._load_model(self.model_json, self.weights)
        self.model = h.model
        print('Loaded model with {} hourglass stacks!'.format(self.num_hg_blocks))

    # ----------------------- PUBLIC METHODS BELOW ----------------------- #

    # Returns np array of predicted heatmaps for a given image and model
    def predict_heatmaps(self, X_batch):
        return np.array(self.model.predict(X_batch)) # output shape is (num_hg_blocks, X_batch_size, 64, 64, 17)

    def visualize_batch(self, X_batch, y_batch, m_batch):
        predicted_heatmaps_batch = self.predict_heatmaps(X_batch)
        for i in range(len(X_batch)):
            X = X_batch[i,]
            y = y_batch[i,]
            m = m_batch[i]
            predicted_heatmaps = predicted_heatmaps_batch[:,i,]
            self._save_stacked_evaluation_heatmaps(X, y, str(m['ann_id']) + '.png', predicted_heatmaps)

    def visualize_heatmaps(self, X_batch, y_batch, img_name_batch):
        pass

    def heatmap_to_COCO_format(self, predicted_hm_batch, metadata_batch):
        list_of_predictions = []
        image_ids = []
        for i, metadata in enumerate(metadata_batch):
            keypoints = self._heatmaps_to_keypoints(predicted_hm_batch[self.num_hg_blocks-1, i, :, :, :])
            metadata = self._undo_bounding_box_transformations(metadata, keypoints)
            list_of_predictions.append(self._create_oks_obj(metadata))
            image_ids.append(metadata['src_set_image_id'])
        return image_ids, list_of_predictions

    def oks_eval(self, image_ids, list_of_predictions, cocoGt):
        cocoDt=cocoGt.loadRes(list_of_predictions)
        annType = "keypoints"
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds = image_ids
        cocoEval.params.catIds = [1] # Person category
        cocoEval.evaluate()
        cocoEval.accumulate()
        print('\nSummary: ')
        cocoEval.summarize()
        return cocoEval.stats

    # This function evaluates PCK@0.2 == Distance between predicted and true joint < 0.2 * torso diameter
    # The PCK_THRESHOLD constant can be updated to adjust this threshold
    # https://github.com/cbsudux/Human-Pose-Estimation-101#percentage-of-correct-key-points---pck
    def pck_eval(self, list_of_predictions):
        f = open(DEFAULT_VAL_ANNOT_PATH)
        data = json.load(f)

        # This function depends on the keypoints order listed in constants COCO_KEYPOINT_LABEL_ARR
        dist_list = []
        correct_keypoints = {
            "nose": 0,
            "left_eye": 0,
            "right_eye": 0,
            "left_ear": 0,
            "right_ear": 0,
            "left_shoulder": 0,
            "right_shoulder": 0,
            "left_elbow": 0,
            "right_elbow": 0,
            "left_wrist": 0,
            "right_wrist": 0,
            "left_hip": 0,
            "right_hip": 0,
            "left_knee": 0,
            "right_knee": 0,
            "left_ankle": 0,
            "right_ankle": 0
        }
        for prediction in list_of_predictions:
            prediction_image_id = prediction['image_id']
            prediction_keypoints = prediction['keypoints']
            for i, ann in enumerate(data['annotations']):
                if data['annotations'][i]['image_id'] == prediction_image_id:
                    annotation_keypoints = data['annotations'][i]['keypoints']
                    prediction_keypoints = np.array(prediction_keypoints)
                    annotation_keypoints = np.array(annotation_keypoints)

                    # Calculate PCK@0.2 threshold for image
                    # Joint at 11 is left hip, Joint at 12 is right hip. Multiply by 3 as each keypoint has (x, y, visibility) to get the array index
                    left_hip_point = np.array(annotation_keypoints[33], annotation_keypoints[34])
                    right_hip_point = np.array(annotation_keypoints[36], annotation_keypoints[37])
                    torso = np.linalg.norm(left_hip_point-right_hip_point)
                    threshold = PCK_THRESHOLD*torso

                    for i in range(NUM_COCO_KEYPOINTS):
                        base = i * NUM_COCO_KP_ATTRBS
                        prediction_point = np.array(prediction_keypoints[base], prediction_keypoints[base+1])
                        annotation_point = np.array(annotation_keypoints[base], annotation_keypoints[base+1])
                        dist = (np.linalg.norm(prediction_point-annotation_point))
                        dist_list.append(dist)

            # Each image may have more than one annotation we need to check
            annotations = int(len(dist_list)/NUM_COCO_KEYPOINTS)

            nose_correct            = False
            left_eye_correct        = False
            right_eye_correct       = False
            left_ear_correct        = False
            right_ear_correct       = False
            left_shoulder_correct   = False
            right_shoulder_correct  = False
            left_elbow_correct      = False
            right_elbow_correct     = False
            left_wrist_correct      = False
            right_wrist_correct     = False
            left_hip_correct        = False
            right_hip_correct       = False
            left_knee_correct       = False
            right_knee_correct      = False
            left_ankle_correct      = False
            right_ankle_correct     = False

            # Append True to correct joint list if distance is below threshold for any annotation
            for j in range(annotations):
                base = j * NUM_COCO_KEYPOINTS
                nose_correct            = nose_correct              or dist_list[0+base]  <= threshold
                left_eye_correct        = left_eye_correct          or dist_list[1+base]  <= threshold
                right_eye_correct       = right_eye_correct         or dist_list[2+base]  <= threshold
                left_ear_correct        = left_ear_correct          or dist_list[3+base]  <= threshold
                right_ear_correct       = right_ear_correct         or dist_list[4+base]  <= threshold
                left_shoulder_correct   = left_shoulder_correct     or dist_list[5+base]  <= threshold
                right_shoulder_correct  = right_shoulder_correct    or dist_list[6+base]  <= threshold
                left_elbow_correct      = left_elbow_correct        or dist_list[7+base]  <= threshold
                right_elbow_correct     = right_elbow_correct       or dist_list[8+base]  <= threshold
                left_wrist_correct      = left_wrist_correct        or dist_list[9+base]  <= threshold
                right_wrist_correct     = right_wrist_correct       or dist_list[10+base] <= threshold
                left_hip_correct        = left_hip_correct          or dist_list[11+base] <= threshold
                right_hip_correct       = right_hip_correct         or dist_list[12+base] <= threshold
                left_knee_correct       = left_knee_correct         or dist_list[13+base] <= threshold
                right_knee_correct      = right_knee_correct        or dist_list[14+base] <= threshold
                left_ankle_correct      = left_ankle_correct        or dist_list[15+base] <= threshold
                right_ankle_correct     = right_ankle_correct       or dist_list[16+base] <= threshold

            # Add one to correct keypoint count if any annotation was below threshold for image
            if nose_correct:            correct_keypoints["nose"]            += 1
            if left_eye_correct:        correct_keypoints["left_eye"]        += 1
            if right_eye_correct:       correct_keypoints["right_eye"]       += 1
            if left_ear_correct:        correct_keypoints["left_ear"]        += 1
            if right_ear_correct:       correct_keypoints["right_ear"]       += 1
            if left_shoulder_correct:   correct_keypoints["left_shoulder"]   += 1
            if right_shoulder_correct:  correct_keypoints["right_shoulder"]  += 1
            if left_elbow_correct:      correct_keypoints["left_elbow"]      += 1
            if right_elbow_correct:     correct_keypoints["right_elbow"]     += 1
            if left_wrist_correct:      correct_keypoints["left_wrist"]      += 1
            if right_wrist_correct:     correct_keypoints["right_wrist"]     += 1
            if left_hip_correct:        correct_keypoints["left_hip"]        += 1
            if right_hip_correct:       correct_keypoints["right_hip"]       += 1
            if left_knee_correct:       correct_keypoints["left_knee"]       += 1
            if right_knee_correct:      correct_keypoints["right_knee"]      += 1
            if left_ankle_correct:      correct_keypoints["left_ankle"]      += 1
            if right_ankle_correct:     correct_keypoints["right_ankle"]     += 1
            dist_list = []

        samples = len(list_of_predictions)
        pck = {k: v/samples for k,v in correct_keypoints.items()}
        print("Percentage of Correct Key Points (PCK)\n")
        print("Nose:            {:.2f}".format(pck["nose"]))
        print("Left Eye:        {:.2f}".format(pck["left_eye"]))
        print("Right Eye:       {:.2f}".format(pck["right_eye"]))
        print("Left Ear:        {:.2f}".format(pck["left_ear"]))
        print("Right Ear:       {:.2f}".format(pck["right_ear"]))
        print("Left Shoulder:   {:.2f}".format(pck["left_shoulder"]))
        print("Right Shoulder:  {:.2f}".format(pck["right_shoulder"]))
        print("Left Elbow:      {:.2f}".format(pck["left_elbow"]))
        print("Right Elbow:     {:.2f}".format(pck["right_elbow"]))
        print("Left Wrist:      {:.2f}".format(pck["left_wrist"]))
        print("Right Wrist:     {:.2f}".format(pck["right_wrist"]))
        print("Left Hip:        {:.2f}".format(pck["left_hip"]))
        print("Right Hip:       {:.2f}".format(pck["right_hip"]))
        print("Left Knee:       {:.2f}".format(pck["left_knee"]))
        print("Right Knee:      {:.2f}".format(pck["right_knee"]))
        print("Left Ankle:      {:.2f}".format(pck["left_ankle"]))
        print("Right Ankle:     {:.2f}".format(pck["right_ankle"]))
        f.close()
        return pck
    # ----------------------- PRIVATE METHODS BELOW ----------------------- #

    # Vertically stack images of different widths
    # https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
    def _vstack_images(self, img_list, interpolation=cv2.INTER_CUBIC):
        # take minimum width
        w_min = min(img.shape[1] for img in img_list)

        # resizing images
        im_list_resize = [cv2.resize(img,
                                     (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                     interpolation=interpolation)
                          for img in img_list]
        # return final image
        return cv2.vconcat(im_list_resize)

    #  Returns np array of stacked ground truth heatmaps for a given image and label
    def _stacked_ground_truth_heatmaps(self, X, y):
        ground_truth_heatmaps = []
        for i in range(NUM_COCO_KEYPOINTS):
            heatmap = y[:,:,i]
            hm = HeatMap.HeatMap(X, heatmap)
            heatmap_array = hm.get_heatmap_array(transparency=0.5)
            ground_truth_heatmaps.append(heatmap_array)
        for i, heatmap in enumerate(ground_truth_heatmaps):
            if(i == 0):
                stacked_ground_truth_heatmaps = ground_truth_heatmaps[0]
            else:
                stacked_ground_truth_heatmaps = np.hstack((stacked_ground_truth_heatmaps, heatmap))
        return stacked_ground_truth_heatmaps

    #  Returns np array of stacked predicted heatmaps
    def _stacked_predict_heatmaps(self, predict_heatmaps):
        for h in range(self.num_hg_blocks):
            stacked_predict_heatmaps = np.array(predict_heatmaps[h, :, :, 0])
            for i in range(NUM_COCO_KEYPOINTS):
                if(i != 0):
                    stacked_predict_heatmaps = np.hstack((stacked_predict_heatmaps, predict_heatmaps[h, :, :, i]))
            if(h == 0):
                stacked_hourglass_heatmaps = np.array(stacked_predict_heatmaps)
            else:
                stacked_hourglass_heatmaps = np.vstack((stacked_hourglass_heatmaps, stacked_predict_heatmaps))
        return stacked_hourglass_heatmaps


    """
    Visualize the set of keypoints on the model image.

    Note, it is assumed that the images are the transformed size for now

    ## Parameters

    keypoints : {list of (x,y,score)}
    """
    def visualize_keypoints(self, X_batch, keypoints, filename):
        # TODO batch functionality

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
        plt.imshow(X_batch)
        plt.savefig(os.path.join(DEFAULT_OUTPUT_BASE_DIR, f'{filename}_saved_scatter.png'), bbox_inches='tight', transparent=False, dpi=300)

    #  Saves to disk stacked predicted heatmaps and stacked ground truth heatmaps and one evaluation image
    def _save_stacked_evaluation_heatmaps(self, X, y, filename, predicted_heatmaps):
        stacked_predict_heatmaps=self._stacked_predict_heatmaps(predicted_heatmaps)
        stacked_ground_truth_heatmaps=self._stacked_ground_truth_heatmaps(X, y)

        # Reshape heatmaps to 3 channels with colour injection, normalize channels to [0,255]
        stacked_predict_heatmaps = cv2.normalize(stacked_predict_heatmaps, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        stacked_predict_heatmaps = cv2.applyColorMap(stacked_predict_heatmaps, cv2.COLORMAP_JET)
        stacked_predict_heatmaps = cv2.normalize(stacked_predict_heatmaps, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        stacked_ground_truth_heatmaps = cv2.cvtColor(stacked_ground_truth_heatmaps, cv2.COLOR_BGRA2RGB)
        stacked_ground_truth_heatmaps = cv2.normalize(stacked_ground_truth_heatmaps, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        heatmap_imgs = []
        heatmap_imgs.append(stacked_predict_heatmaps)
        heatmap_imgs.append(stacked_ground_truth_heatmaps)

        # Resize and vertically stack heatmap images
        img_v_resize = self._vstack_images(heatmap_imgs)

        cv2.imwrite(os.path.join(self.output_sub_dir,filename), img_v_resize)

    # Resources for heatmaps to keypoints
    # https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/blob/eddf0ae15715a88d7859847cfff5f5092b260ae1/src/eval/heatmap_process.py#L5
    # https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/56707252501c73b2bf2aac8fff3e22760fd47dca/hourglass/postprocess.py#L17

    ### Returns np array of predicted keypoints from one image's heatmaps
    def _heatmaps_to_keypoints(self, heatmaps, threshold=HM_TO_KP_THRESHOLD):
        keypoints = list()
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
            y, x = np.where(peaks == peaks.max())
            if int(x[0]) > 0 and int(y[0]) > 0:
                keypoints.append((int(x[0]), int(y[0]), peaks[y[0], x[0]]))
            else:
                keypoints.append((0, 0, 0))
        # Turn keypoints into np array
        keypoints = np.array(keypoints)
        return keypoints

    def _non_max_supression(self, plain, threshold, windowSize=3):
        # Clear values less than threshold
        under_thresh_indices = plain < threshold
        plain[under_thresh_indices] = 0
        return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))

    """
        Parameters
        ----------
        metadata : object
        should be metadata associated to a single image

        untransformed_x : int
        x coordinate to
    """
    def _undo_x(self, metadata, untransformed_x):
      predicted_x = round(untransformed_x * metadata['cropped_width'] / metadata['input_dim'][0] + metadata['anchor_x'])
      return int(predicted_x)

    """
        Parameters
        ----------
        metadata : object
        should be metadata associated to a single image

        untransformed_y : int
        x coordinate to
    """
    def _undo_y(self, metadata, untransformed_y):
      predicted_y = round(untransformed_y * metadata['cropped_height'] / metadata['input_dim'][1] + metadata['anchor_y'])
      return int(predicted_y)

    """
        Parameters
        ----------
        metadata : object
        should be metadata associated to a single image

        untransformed_predictions : list
        a list of precitions that need to be transformed
        Example:  [1,2,0,1,4,666,32...]
    """
    def _undo_bounding_box_transformations(self, metadata, untransformed_predictions):
        untransformed_predictions = np.array(untransformed_predictions).flatten()
        predicted_labels = []
        list_of_scores = []
        for i in range(len(untransformed_predictions)):
            if i % 3 == 0: # is an x-coord
                predicted_labels.append(self._undo_x(metadata, untransformed_predictions[i]))
            elif i % 3 == 1: # is a y-coord
                predicted_labels.append(self._undo_y(metadata, untransformed_predictions[i]))
            elif i % 3 == 2: # is a confidence score
                if(untransformed_predictions[i] == 0): # this keypoint is not predicted
                    predicted_labels[i-1] = 0 # Set y value to 0
                    predicted_labels[i-2] = 0 # Set x value to 0
                    predicted_labels.append(0) # set visibility to 0
                else:
                    predicted_labels.append(1) # set visibility to 1
                    list_of_scores.append(untransformed_predictions[i])
        metadata['predicted_labels'] = predicted_labels
        metadata['score'] = float(np.mean(np.array(list_of_scores)))
        return metadata

    def _create_oks_obj(self, metadata):
        oks_obj = {}
        oks_obj["image_id"] = int(metadata['src_set_image_id'])
        oks_obj["category_id"] = 1
        oks_obj["keypoints"] = metadata['predicted_labels']
        oks_obj["score"] = float(metadata['score'])
        return oks_obj

# ----------------------- End of Class -----------------------

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

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

        self.model_json, self.weights, _ = util.find_resume_json_weights_str(model_base_dir, model_sub_dir, epoch)

        if not os.path.exists(self.output_sub_dir):
            os.makedirs(self.output_sub_dir)

        self.num_hg_blocks = int(re.match(r'.*stacks_([\d]+).*$',self.model_json).group(1))
        h = hourglass.HourglassNet(NUM_COCO_KEYPOINTS,self.num_hg_blocks,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
        h._load_model(self.model_json, self.weights)
        self.model = h.model
        print('Loaded model with {} hourglass stacks!'.format(self.num_hg_blocks))

    # ----------------------- PUBLIC METHODS BELOW ----------------------- #

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
    This method has been deprecated in favour of the `visualizeHeatmaps` method in `evaluation_wrapper`
    """
    def visualize_batch(self, X_batch, y_batch, m_batch):
        raise DeprecationWarning('visualize_batch has been deprecated in favour of the `visualizeHeatmaps` method in `evaluation_wrapper`')
        # predicted_heatmaps_batch = self.predict_heatmaps(X_batch)

        # img_id_batch = [m['ann_id'] for m in m_batch]

        # self.visualize_heatmaps(X_batch, y_batch, img_id_batch, predicted_heatmaps_batch)

    """
    Visualize the set of stacked heatmap predictions.

    ## Parameters

    X_batch : {list of ndarrays}
        A list of images which were used as input to the model

    y_batch : {list of ndarrays}
        A list of ground truth heatmaps from a single hourglass layer

    img_id_batch : {list of strings}
        A list of image names. These should not contain the extension, epoch, or type. (Purely image ID)

    predicted_heatmaps_batch : {list of ndarrays}
        A list of heatmap predictions from the model from all hourglass layers
    """
    def visualize_heatmaps(self, X_batch, y_batch, img_id_batch, predicted_heatmaps_batch):
        # Clear existing plots
        plt.clf()

        for i in range(len(X_batch)):
            X = X_batch[i,]
            y = y_batch[i,]
            img_id = img_id_batch[i]
            name = f'{OUTPUT_STACKED_HEATMAP}_{img_id}_{self.epoch}.png'

            predicted_heatmaps = predicted_heatmaps_batch[:,i,]
            self._save_stacked_evaluation_heatmaps(X, y, name, predicted_heatmaps)

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
    def visualize_keypoints(self, X_batch, keypoints_batch, img_id_batch, show_skeleton=True):
        # Clear existing plots
        plt.clf()

        for i in range(len(X_batch)):
            X = X_batch[i]
            keypoints = keypoints_batch[i]
            img_id = img_id_batch[i]
            name = f'{img_id}_{self.epoch}.png'
            if show_skeleton:
                name = f'{OUTPUT_SKELETON}_{name}'

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
            plt.imshow(X)
            plt.savefig(os.path.join(self.output_sub_dir, name), bbox_inches='tight', transparent=False, dpi=300)
            plt.close()

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

    def heatmap_to_COCO_format(self, predicted_hm_batch, metadata_batch):
        list_of_predictions = []
        image_ids = []
        for i, metadata in enumerate(metadata_batch):
            keypoints = self.heatmaps_to_keypoints(predicted_hm_batch[self.num_hg_blocks-1, i, :, :, :])
            metadata = self._undo_bounding_box_transformations(metadata, keypoints)
            list_of_predictions.append(self._create_oks_obj(metadata))
            image_ids.append(metadata['src_set_image_id'])
        return image_ids, list_of_predictions

    def oks_eval(self, image_ids, list_of_predictions, cocoGt):
        cocoDt=cocoGt.loadRes(list_of_predictions)

        # Convert keypoint predictions to int type
        for i in range(len(list_of_predictions)):
            list_of_predictions[i]["keypoints"] = list_of_predictions[i]["keypoints"].astype('int')

        annType = "keypoints"
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds = image_ids
        cocoEval.params.catIds = [1] # Person category
        cocoEval.evaluate()
        cocoEval.accumulate()
        print('\nSummary: ')
        cocoEval.summarize()
        stats = cocoEval.stats
        oks = {
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]': stats[0],
            'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]': stats[1],
            'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]': stats[2],
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]': stats[3],
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]': stats[4],
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]': stats[5],
            'Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ]': stats[6],
            'Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ]': stats[7],
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]': stats[8],
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]': stats[9]
        }
        return oks

    # This function evaluates PCK@0.2 == Distance between predicted and true joint < 0.2 * torso diameter
    # The PCK_THRESHOLD constant can be updated to adjust this threshold
    # https://github.com/cbsudux/Human-Pose-Estimation-101#percentage-of-correct-key-points---pck
    def pck_eval(self, list_of_predictions):
        # This function depends on the keypoints order listed in constants COCO_KEYPOINT_LABEL_ARR

        f = open(DEFAULT_VAL_ANNOT_PATH)
        data = json.load(f)

        correct_keypoints = {
            "nose":             0,
            "left_eye":         0,
            "right_eye":        0,
            "left_ear":         0,
            "right_ear":        0,
            "left_shoulder":    0,
            "right_shoulder":   0,
            "left_elbow":       0,
            "right_elbow":      0,
            "left_wrist":       0,
            "right_wrist":      0,
            "left_hip":         0,
            "right_hip":        0,
            "left_knee":        0,
            "right_knee":       0,
            "left_ankle":       0,
            "right_ankle":      0
        }

        num_anns = len(data['annotations'])

        for prediction in list_of_predictions:
            prediction_image_id = prediction['image_id']
            prediction_ann_id = prediction['ann_id']
            prediction_keypoints = prediction['keypoints']

            dist_list = []

            # Find the annotation ID we were predicting for
            for i in range(num_anns):
                if data['annotations'][i]['id'] == prediction_ann_id:
                    annotation_keypoints = data['annotations'][i]['keypoints']
                    prediction_keypoints = np.array(prediction_keypoints)
                    annotation_keypoints = np.array(annotation_keypoints)

                    # Calculate PCK@0.2 threshold for image
                    # TODO figure out what to do if a hip or ear isn't present
                    threshold = DEFAULT_PCK_THRESHOLD

                    # If both hips are present
                    # Joint at 11 is left hip, Joint at 12 is right hip. Multiply by 3 as each keypoint has (x, y, visibility) to get the array index
                    # Check visibility flags for both hip joints
                    if annotation_keypoints[35] > 0 and annotation_keypoints[38] > 0:
                        left_hip_point = np.array(annotation_keypoints[33], annotation_keypoints[34])
                        right_hip_point = np.array(annotation_keypoints[36], annotation_keypoints[37])
                        torso = np.linalg.norm(left_hip_point-right_hip_point)
                        threshold = PCK_THRESHOLD*torso
                    # Use head threshold if no torso exists
                    # Joint at 1 is left ear, Joint at 2 is right ear. Multiply by 3 as each keypoint has (x, y, visibility) to get the array index
                    # Check visibility flags for both ear joints
                    elif annotation_keypoints[5] > 0 and annotation_keypoints[8] > 0:
                        left_ear_point = np.array(annotation_keypoints[3], annotation_keypoints[4])
                        right_ear_point = np.array(annotation_keypoints[6], annotation_keypoints[7])
                        head = np.linalg.norm(left_ear_point-right_ear_point)
                        threshold = PCK_THRESHOLD*head

                    for i in range(NUM_COCO_KEYPOINTS):
                        base = i * NUM_COCO_KP_ATTRBS
                        prediction_point = np.array(prediction_keypoints[base], prediction_keypoints[base+1])
                        annotation_point = np.array(annotation_keypoints[base], annotation_keypoints[base+1])
                        dist = np.linalg.norm(prediction_point-annotation_point)
                        dist_list.append(dist)

                    break

            # True to correct joint if distance is below threshold for any annotation
            # Add one to correct keypoint count if any annotation was below threshold for image
            correct_keypoints["nose"]            += int(dist_list[0]  <= threshold)
            correct_keypoints["left_eye"]        += int(dist_list[1]  <= threshold)
            correct_keypoints["right_eye"]       += int(dist_list[2]  <= threshold)
            correct_keypoints["left_ear"]        += int(dist_list[3]  <= threshold)
            correct_keypoints["right_ear"]       += int(dist_list[4]  <= threshold)
            correct_keypoints["left_shoulder"]   += int(dist_list[5]  <= threshold)
            correct_keypoints["right_shoulder"]  += int(dist_list[6]  <= threshold)
            correct_keypoints["left_elbow"]      += int(dist_list[7]  <= threshold)
            correct_keypoints["right_elbow"]     += int(dist_list[8]  <= threshold)
            correct_keypoints["left_wrist"]      += int(dist_list[9]  <= threshold)
            correct_keypoints["right_wrist"]     += int(dist_list[10] <= threshold)
            correct_keypoints["left_hip"]        += int(dist_list[11] <= threshold)
            correct_keypoints["right_hip"]       += int(dist_list[12] <= threshold)
            correct_keypoints["left_knee"]       += int(dist_list[13] <= threshold)
            correct_keypoints["right_knee"]      += int(dist_list[14] <= threshold)
            correct_keypoints["left_ankle"]      += int(dist_list[15] <= threshold)
            correct_keypoints["right_ankle"]     += int(dist_list[16] <= threshold)

        samples = len(list_of_predictions)
        pck = {k: v/samples for k,v in correct_keypoints.items()}
        pck['avg'] = sum(pck.values())/len(pck)
        print("Percentage of Correct Key Points (PCK)\n")
        print("Average PCK:     {:.2f}".format(pck['avg']))
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

        cv2.imwrite(os.path.join(self.output_sub_dir, filename), img_v_resize)

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
      return round(predicted_x)

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
      return round(predicted_y)

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
        untransformed_predictions = untransformed_predictions.flatten()

        predicted_labels = np.zeros(NUM_COCO_KEYPOINTS * NUM_COCO_KP_ATTRBS)
        list_of_scores = np.zeros(NUM_COCO_KEYPOINTS)

        for i in range(NUM_COCO_KEYPOINTS):
            base = i * NUM_COCO_KP_ATTRBS

            x = untransformed_predictions[base]
            y = untransformed_predictions[base + 1]
            conf = untransformed_predictions[base + 2]

            if conf == 0:
                # this keypoint is not predicted
                x_new, y_new, vis_new = 0, 0, 0
            else:
                x_new = self._undo_x(metadata, x)
                y_new = self._undo_y(metadata, y)
                vis_new = 1
                list_of_scores[i] = conf

            predicted_labels[base]     = x_new
            predicted_labels[base + 1] = y_new
            predicted_labels[base + 2] = vis_new

        metadata['predicted_labels'] = predicted_labels
        metadata['score'] = float(np.mean(list_of_scores))
        return metadata

    def _create_oks_obj(self, metadata):
        oks_obj = {}
        oks_obj["image_id"] = int(metadata['src_set_image_id'])
        oks_obj["ann_id"] = int(metadata['ann_id'])
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

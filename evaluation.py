import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from constants import *
import HeatMap  # https://github.com/LinShanify/HeatMap
import util
import hourglass



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

        self.model_json, self.weights = util.find_resume_json_weights_str(model_base_dir, model_sub_dir, epoch)
        self.num_hg_blocks = int(re.match(r'.*stacks_(\d\d)_.*',self.model_json).group(1))
        h = hourglass.HourglassNet(NUM_COCO_KEYPOINTS,self.num_hg_blocks,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
        h._load_model(self.model_json, self.weights)
        self.model = h.model
        print('Loaded model with {} hourglass stacks!'.format(self.num_hg_blocks))


    # Vertically stack images of different widths
    # https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
    def _vstack_images(self, img_list, interpolation=cv2.INTER_CUBIC):
        # take minimum width 
        w_min = min(img.shape[1] for img in img_list) 
        
        # resizing images 
        im_list_resize = [cv2.resize(img, 
                        (w_min, int(img.shape[0] * w_min / img.shape[1])), 
                                    interpolation = interpolation) 
                        for img in img_list] 
        # return final image 
        return cv2.vconcat(im_list_resize) 

    # Returns np array of predicted heatmaps for a given image and model
    def predict_heatmaps(self, X_batch):
        predicted_heatmaps = np.array(self.model.predict(X_batch)) # output shape is (num_hg_blocks, X_batch_size, 64, 64, 17)
        for i in range(len(X_batch)):
            hm = predicted_heatmaps[:,i,]
            print('model prediction metrics: ')
            predict_mean = np.mean(hm)
            predict_max = np.max(hm)
            predict_min = np.min(hm)
            predict_var = np.var(hm.flatten())
            print('Mean: {:0.6e}\t Max: {:e}\t Min: {:e}\t Variance: {:e}'.format(predict_mean, predict_max, predict_min, predict_var))
        # # TODO: is this how we want to normalize output? Should we instead enforce model output to be normalized?
        # NOTE: this normalization gets applied in the stacking function
        # normalized_heatmaps = predict_heatmaps / predict_max
        # normalized_heatmaps = normalized_heatmaps - predict_min
        # normalized_heatmaps = normalized_heatmaps / np.max(normalized_heatmaps)
        # return normalized_heatmaps
        return predicted_heatmaps

    #  Returns np array of stacked ground truth heatmaps for a given image and label
    def stacked_ground_truth_heatmaps(self, X, y):
        ground_truth_heatmaps = []
        for i in range(NUM_COCO_KEYPOINTS):
            heatmap = y[:,:,i]
            hm = HeatMap.HeatMap(X,heatmap)
            heatmap_array = hm.get_heatmap_array(transparency=0.5)
            ground_truth_heatmaps.append(heatmap_array)
        for i, heatmap in enumerate(ground_truth_heatmaps):
            if(i == 0):
                stacked_ground_truth_heatmaps = ground_truth_heatmaps[0]
            else:
                stacked_ground_truth_heatmaps = np.hstack((stacked_ground_truth_heatmaps, heatmap))
        return stacked_ground_truth_heatmaps

    #  Returns np array of stacked predicted heatmaps
    def stacked_predict_heatmaps(self, predict_heatmaps):
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

    def visualize_batch(self, X_batch, y_batch, m_batch):
        predicted_heatmaps_batch = self.predict_heatmaps(X_batch)
        for i in range(len(X_batch)):
            X = X_batch[i,]
            y = y_batch[i,]
            m = m_batch[i]
            predicted_heatmaps = predicted_heatmaps_batch[:,i,]
            self.save_stacked_evaluation_heatmaps(X, y, m, predicted_heatmaps)


    #  Saves to disk stacked predicted heatmaps and stacked ground truth heatmaps and one evaluation image
    def save_stacked_evaluation_heatmaps(self, X, y, m, predicted_heatmaps):
        stacked_predict_heatmaps=self.stacked_predict_heatmaps(predicted_heatmaps)
        stacked_ground_truth_heatmaps=self.stacked_ground_truth_heatmaps(X, y)
        
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

        filename = str(m['ann_id']) + '.png'
        cv2.imwrite(os.path.join(self.output_sub_dir,filename), img_v_resize) 
    

"""
Runs the model for any general file. This aims to extend the DataGenerator output format for arbitrary images

## Parameters:
img_path : {string-typed} path to image
    Note this image must be square, and centered around the person you wish to retrieve predictions for.

num_hg_blocks : {int} number of hourglass blocks to generate dummy ground truth data for

x,y,w,h : {int or float} optional bounding box info, anchored at top left of image
"""
def load_and_preprocess_img(img_path, num_hg_blocks, x=None, y=None, w=None, h=None):
    img = Image.open(img_path).convert('RGB')

    if x is None or y is None or w is None or h is None:
        # If any of the parameters are missing, crop a square area from top left of image
        smaller_dim = img.size[0] if img.size[0] <= img.size[1] else img.size[1]
        bbox = [0, 0, smaller_dim, smaller_dim]
    else:
        # If a bounding box is provided, use it
        bbox = [x, y, w, h]
    
    bbox = np.array(bbox)
    
    cropped_img = img.crop(box=bbox)
    cropped_width, cropped_height = cropped_img.size
    new_img = cv2.resize(np.array(cropped_img), INPUT_DIM,
                        interpolation=cv2.INTER_LINEAR)
    
    # Add a 'batch' axis
    X_batch = np.expand_dims(new_img.astype('float'), axis=0)

    # Add dummy heatmap "ground truth", duplicated 'num_hg_blocks' times
    y_batch = [np.zeros((1, *(OUTPUT_DIM), NUM_COCO_KEYPOINTS)) for _ in range(num_hg_blocks)]

    # Normalize input image
    X_batch /= 255
    return X_batch, y_batch

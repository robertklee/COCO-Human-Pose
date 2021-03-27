import HeatMap # https://github.com/LinShanify/HeatMap
from constants import *

import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import os

class Evaluation():

    def __init__(self, model_json, weights, h_net, base_dir=DEFAULT_MODEL_BASE_DIR,sub_dir=''): 
        self.sub_dir = sub_dir
        self.model_json = os.path.join(base_dir,sub_dir,model_json)         # json of model to be evaluated
        self.weights = os.path.join(base_dir,sub_dir,weights)          # weights of model to be evaluated
        self.num_hg_blocks = int(re.match(r'.*stacks_(\d\d)_.*',model_json).group(1))
        h_net._load_model(self.model_json, self.weights)
        self.model = h_net.model

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
    def predict_heatmaps(self, X):
        X = np.expand_dims(X, axis=0) # add "batch" dimension of 1 because predict needs shape (1, 256, 256, 3)
        predict_heatmaps = self.model.predict(X)
        predict_heatmaps = np.array(predict_heatmaps) # output shape is (num_hg_blocks, 1, 64, 64, 17)
        return predict_heatmaps

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
            stacked_predict_heatmaps = np.array(predict_heatmaps[h, 0, :, :, 0])
            for i in range(NUM_COCO_KEYPOINTS):
                if(i != 0):
                    stacked_predict_heatmaps = np.hstack((stacked_predict_heatmaps, predict_heatmaps[h, 0, :, :, i]))
            if(h == 0):
                stacked_hourglass_heatmaps = np.array(stacked_predict_heatmaps)
            else:
                stacked_hourglass_heatmaps = np.vstack((stacked_hourglass_heatmaps, stacked_predict_heatmaps))
        return stacked_hourglass_heatmaps

    #  Saves to disk stacked predicted heatmaps and stacked ground truth heatmaps and one evaluation image
    def save_stacked_evaluation_heatmaps(self, X, y, filename):
        predict_heatmaps=self.predict_heatmaps(X)
        stacked_predict_heatmaps=self.stacked_predict_heatmaps(predict_heatmaps)
        stacked_ground_truth_heatmaps=self.stacked_ground_truth_heatmaps(X, y)
        
        # Reshape heatmaps to 3 channels, normalize channels to [0,255]
        stacked_predict_heatmaps = cv2.cvtColor(stacked_predict_heatmaps, cv2.COLOR_GRAY2RGB)
        stacked_predict_heatmaps = cv2.normalize(stacked_predict_heatmaps, None, alpha=0, beta=255.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        stacked_ground_truth_heatmaps = cv2.cvtColor(stacked_ground_truth_heatmaps, cv2.COLOR_BGRA2RGB)
        stacked_ground_truth_heatmaps = cv2.normalize(stacked_ground_truth_heatmaps, None, alpha=0, beta=255.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        heatmap_imgs = []
        heatmap_imgs.append(stacked_predict_heatmaps)
        heatmap_imgs.append(stacked_ground_truth_heatmaps)

        # Resize and vertically stack heatmap images
        img_v_resize = self._vstack_images(heatmap_imgs) 
        
        cv2.imwrite(filename, img_v_resize) 

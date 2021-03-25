from submodules.HeatMap import HeatMap # https://github.com/LinShanify/HeatMap
from constants import *

import matplotlib.pyplot as plt
import numpy as np
import cv2

class Evaluation():

    def __init__(self, model_json, weights, df, num_hg_blocks, batch_size=1):
        self.model_json = model_json              # json of model to be evaluated
        self.weights = weights          # weights of model to be evaluated
        self.df = df                    # df of the the annotations we want
        self.num_hg_blocks = num_hg_blocks
        self.batch_size = batch_size

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
    def predict_heatmaps(self, h, X):
        h._load_model(self.model_json, self.weights)

        X = X.reshape(1, 256, 256, 3) # predict needs shape (1, 256, 256, 3)
        predict_heatmaps = h.model.predict(X)
        predict_heatmaps = np.array(predict_heatmaps) # output shape is (num_hg_blocks, 1, 64, 64, 17)
        return predict_heatmaps

    #  Returns np array of stacked ground truth heatmaps for a given image and label
    def stacked_ground_truth_heatmaps(self, X, y):
        ground_truth_heatmaps = []
        for i in range(NUM_COCO_KEYPOINTS):
            heatmap = y[:,:,i]
            hm = HeatMap(X,heatmap)
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

    #  Saves stacked predicted heatmaps and stacked ground truth heatmaps in one evaluation image
    def save_stacked_evaluation_heatmaps(self, stacked_predict_heatmaps_file, stacked_ground_truth_heatmaps_file, filename):
        heatmap_imgs = []
        heatmap_imgs.append(cv2.imread(stacked_predict_heatmaps_file))
        heatmap_imgs.append(cv2.imread(stacked_ground_truth_heatmaps_file))

        # Resize and vertically stack heatmap images
        img_v_resize = self._vstack_images(heatmap_imgs) 
        
        cv2.imwrite(filename, img_v_resize) 

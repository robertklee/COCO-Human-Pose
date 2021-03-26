import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from constants import *
from HeatMap import HeatMap  # https://github.com/LinShanify/HeatMap


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

        X = np.expand_dims(X, axis=0) # add "batch" dimension of 1 because predict needs shape (1, 256, 256, 3)
        predict_heatmaps = h.model.predict(X)
        predict_heatmaps = np.array(predict_heatmaps) # output shape is (num_hg_blocks, 1, 64, 64, 17)

        print('model prediction metrics: ')
        predict_mean = np.mean(predict_heatmaps)
        predict_max = np.max(predict_heatmaps)
        predict_min = np.min(predict_heatmaps)
        predict_var = np.var(predict_heatmaps.flatten())
        print('Mean: {:0.6e}\t Max: {:e}\t Min: {:e}\t Variance: {:e}'.format(predict_mean, predict_max, predict_min, predict_var))
        normalized_heatmaps = predict_heatmaps / predict_max
        normalized_heatmaps = normalized_heatmaps - predict_min
        normalized_heatmaps = predict_heatmaps / np.max(predict_heatmaps)

        return normalized_heatmaps

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

    #  Saves to disk stacked predicted heatmaps and stacked ground truth heatmaps and one evaluation image
    def save_stacked_evaluation_heatmaps(self, h, X, y, stacked_predict_heatmaps_file, stacked_ground_truth_heatmaps_file, filename):
        predict_heatmaps=self.predict_heatmaps(h, X)
        stacked_predict_heatmaps=self.stacked_predict_heatmaps(predict_heatmaps)
        stacked_ground_truth_heatmaps=self.stacked_ground_truth_heatmaps(X, y)

        # Save stacked images to disk
        plt.imsave(stacked_predict_heatmaps_file, stacked_predict_heatmaps)
        plt.imsave(stacked_ground_truth_heatmaps_file, stacked_ground_truth_heatmaps)
        filename = filename
        
        heatmap_imgs = []
        heatmap_imgs.append(cv2.imread(stacked_predict_heatmaps_file))
        heatmap_imgs.append(cv2.imread(stacked_ground_truth_heatmaps_file))

        # Resize and vertically stack heatmap images
        img_v_resize = self._vstack_images(heatmap_imgs) 
        
        cv2.imwrite(filename, img_v_resize) 
    

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
    X_batch = np.expand_dims(new_img, axis=0)

    # Add dummy heatmap "ground truth", duplicated 'num_hg_blocks' times
    y_batch = [np.zeros((1, *(OUTPUT_DIM), NUM_COCO_KEYPOINTS)) for _ in range(num_hg_blocks)]

    return X_batch, y_batch

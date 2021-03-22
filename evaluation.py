from submodules.HeatMap import HeatMap # https://github.com/LinShanify/HeatMap
from constants import *
# from util import *

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

class Evaluation():

    def __init__(self, model, weights, df, num_hg_blocks, batch_size=1):
        self.model = model              # json of model to be evaluated
        self.weights = weights          # weights of model to be evaluated
        self.df = df                    # df of the the annotations we want
        self.num_hg_blocks = num_hg_blocks
        self.batch_size = batch_size

    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    def hstack_images(self, images, filename):
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

        new_im.save(filename)

    # https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
    def vstack_images(self, img_list, interpolation=cv2.INTER_CUBIC):
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
        h._load_model(self.model, self.weights)

        X = X.reshape(1, 256, 256, 3) # predict needs shape (1, 256, 256, 3)
        predict_heatmaps = h.model.predict(X)
        predict_heatmaps = np.array(predict_heatmaps) # output shape is (num_hg_blocks, 1, 64, 64, 17)
        return predict_heatmaps

    #  Returns filename of saved stacked ground truth heatmaps and saves all ground truth heatmaps
    def save_stacked_ground_truth_heatmaps(self, X, y):
        ground_truth_heatmaps = []
        for i in range(NUM_COCO_KEYPOINTS):
            heatmap = y[:,:,i]
            hm = HeatMap(X,heatmap)
            hm.save(f'ground_truth_heatmap{i}','png', transparency=0.5)
            ground_truth_heatmaps.append(f'ground_truth_heatmap{i}.png')

        images = [Image.open(x) for x in ground_truth_heatmaps]
        filename = 'stacked_ground_truth_heatmaps.png'
        self.hstack_images(images, filename)
        return filename

    #  Returns list of saved stacked predicted heatmaps and saves all predicted heatmaps
    def save_stacked_predict_heatmaps(self, predict_heatmaps):
        stacked_predict_heatmaps = []
        for h in range(self.num_hg_blocks):
            heatmaps = []
            for i in range(NUM_COCO_KEYPOINTS):
                plt.figure(i)
                plt.imsave(f'hourglass{h}_heatmap{i}.png', predict_heatmaps[h, 0, :, :, i])
                heatmaps.append(f'hourglass{h}_heatmap{i}.png')

            images = [Image.open(x) for x in heatmaps]
            self.hstack_images(images, f'stacked_heatmaps_hourglass{h}.png')
            stacked_predict_heatmaps.append(f'stacked_heatmaps_hourglass{h}.png')

        return stacked_predict_heatmaps

    #  Saves stacked predicted heatmaps and stacked ground truth heatmaps in one evaluation image
    def save_stacked_evaluation_heatmaps(self, stacked_predict_heatmaps, stacked_ground_truth_heatmaps):
        heatmaps = []
        for x in stacked_predict_heatmaps:
            heatmaps.append(x)
        heatmaps.append(stacked_ground_truth_heatmaps)
        # Stack each hourglass and ground truth heatmaps into heatmap_evaluation.png
        heatmap_imgs = []
        for x in heatmaps:
            heatmap_imgs.append(cv2.imread(x))

        # Resize and vertically stack heatmap images
        img_v_resize = self.vstack_images(heatmap_imgs) 
        
        cv2.imwrite('heatmap_evaluation.png', img_v_resize) 

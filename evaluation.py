# %% Evaluation Imports
import argparse
from hourglass import HourglassNet
from data_generator import DataGenerator
from submodules.HeatMap import HeatMap # https://github.com/LinShanify/HeatMap
from constants import *
from util import *

import time
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
import glob
import cv2

# %% Declare function to horizontally stack images
# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
def stack_images(images, filename):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(filename)

# %% Declare function to resize and vertically stack images
# https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
def vconcat_resize(img_list, interpolation= cv2.INTER_CUBIC):
    # take minimum width 
    w_min = min(img.shape[1] for img in img_list) 
      
    # resizing images 
    im_list_resize = [cv2.resize(img, 
                      (w_min, int(img.shape[0] * w_min / img.shape[1])), 
                                 interpolation = interpolation) 
                      for img in img_list] 
    # return final image 
    return cv2.vconcat(im_list_resize) 

# %% Declare function to predict heatmaps
def predict(hourglass_num, model_json, model_weights, subset):
    hgnet = HourglassNet(num_classes=NUM_COCO_KEYPOINTS, num_stacks=hourglass_num, num_channels=NUM_CHANNELS, inres=INPUT_DIM,
                            outres=OUTPUT_DIM)
    hgnet._load_model(model_json, model_weights)

    train_df, val_df = hgnet.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH, DEFAULT_VAL_ANNOT_PATH, subset)

    generator = DataGenerator(
        df=val_df,
        base_dir=DEFAULT_TRAIN_IMG_PATH,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        num_hg_blocks=DEFAULT_NUM_HG,
        shuffle=False,  
        batch_size=1,
        online_fetch=False)

    # Select image to predict heatmaps
    X_batch, y_stacked = generator[168]
    y_batch = y_stacked[0] # take first hourglass section
    X, y = X_batch[0], y_batch[0] # take first example of batch

    image_list = []
    # Save ground truth heatmaps
    for i in range(NUM_COCO_KEYPOINTS):
        heatmap = y[:,:,i]
        hm = HeatMap(X,heatmap)
        hm.save(f'ground_truth_heatmap{i}','png', transparency=0.5)
        image_list.append(f'ground_truth_heatmap{i}.png')

    images = [Image.open(x) for x in image_list]
    stack_images(images, 'stacked_ground_truth_heatmaps.png')

    X = X.reshape(1, 256, 256, 3) # predict needs shape (1, 256, 256, 3)
    output = hgnet.model.predict(X)
    output = np.array(output) # output shape is (hourglass_num, 1, 64, 64, 17)

    # Save output heatmaps
    result_list = []
    for h in range(hourglass_num):
        image_list = []
        for i in range(NUM_COCO_KEYPOINTS):
            plt.figure(i)
            plt.imsave(f'hourglass{h}_heatmap{i}.png', output[h, 0, :, :, i])
            image_list.append(f'hourglass{h}_heatmap{i}.png')

        images = [Image.open(x) for x in image_list]
        stack_images(images, f'stacked_heatmaps_hourglass{h}.png')
        result_list.append(f'stacked_heatmaps_hourglass{h}.png')

    result_list.append('stacked_ground_truth_heatmaps.png')

    # Stack each hourglass and ground truth heatmaps into evaluation.png
    result_imgs = []
    for x in result_list:
        result_imgs.append(cv2.imread(x))

    # Resize and vertically stack heatmap images
    img_v_resize = vconcat_resize(result_imgs) 
    
    # Save the output heatmaps with ground truth
    cv2.imwrite('evaluation.png', img_v_resize) 

# %% Declare function to process args
def process_args():
    argparser = argparse.ArgumentParser(description='Evaluation parameters')
    argparser.add_argument('--model-json',
                        default=None,
                        help='Model architecture for re-loading weights to evaluate')
    argparser.add_argument('--model-weights',
                        default=None,
                        help='Model weights file to evaluate')
    argparser.add_argument('--model-subdir',
                        default=None,
                        help='Subdirectory containing evaluation architecture json and weights') 
    argparser.add_argument('--subset',
                        type=float,
                        default=1.0,
                        help='fraction of train set to train on, default 1.0')
    argparser.add_argument('--hourglass',
                        type=int,
                        default=DEFAULT_NUM_HG,
                        help='number of hourglass blocks')
    # Convert string arguments to appropriate type
    args = argparser.parse_args()

     # Validate arguments
    assert (args.subset > 0 and args.subset <= 1.0), "Subset must be fraction between 0 and 1.0"

    if args.model_subdir:
        # Automatically locate architecture json and model weights
        find_resume_json_weights(args)
    
    assert args.model_json is not None and args.model_weights is not None, \
        "Model evaluation enabled, but no parameters received for: --model-subdir, or both --model-json and --model-weights"

    return args

# %% Declare main function
if __name__ == "__main__":
    args = process_args()

    print("\n\nModel evaluation start: {}\n".format(time.ctime()))
    evaluation_start = time.time()

    predict(hourglass_num=args.hourglass, model_json=args.model_json, model_weights=args.model_weights, subset=args.subset)

    print("\n\nModel evaluation end:   {}\n".format(time.ctime()))
    evaluation_end = time.time()

    evaluation_time = evaluation_end - evaluation_start
    print("Total Model evaluation time: {}".format(str(timedelta(seconds=evaluation_time))))

import argparse
from hourglass import HourglassNet
from data_generator import DataGenerator
from constants import *
from util import *

import time
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import cv2

def predict(hourglass, model_json, model_weights, subset):
    hgnet = HourglassNet(num_classes=NUM_COCO_KEYPOINTS, num_stacks=hourglass, num_channels=NUM_CHANNELS, inres=INPUT_DIM,
                            outres=OUTPUT_DIM)
    hgnet._load_model(model_json, model_weights)

    val_df = hgnet.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH, DEFAULT_VAL_ANNOT_PATH, subset)

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
    X = X_batch[0] # take first example of batch

    X = X.reshape(1, 256, 256, 3) # predict needs shape (1, 256, 256, 3)
    output = hgnet.model.predict(X)
    output = np.array(output) # output shape is (hourglass_num, 1, 64, 64, 17)

    # Save output heatmaps
    for i in range(NUM_COCO_KEYPOINTS):
        plt.figure(i)
        plt.imsave(f'image{i}.png', output[3, 0, :, :, i])

    # Overlay heatmaps
    # https://automaticaddison.com/how-to-blend-multiple-images-using-opencv/
    # Import all image files with image*.png
    files = glob.glob ("image*.png")
    image_data = []
    for my_file in files:
        this_image = cv2.imread(my_file, 1)
        image_data.append(this_image)
    
    # Calculate blended image
    dst = image_data[0]
    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(image_data[i], alpha, dst, beta, 0.0)
 
    # Save blended heatmap image
    cv2.imwrite('result.png', dst)

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

if __name__ == "__main__":
    args = process_args()

    print("\n\nModel evaluation start: {}\n".format(time.ctime()))
    evaluation_start = time.time()

    predict(hourglass=args.hourglass, model_json=args.model_json, model_weights=args.model_weights, subset=args.subset)

    print("\n\nModel evaluation end:   {}\n".format(time.ctime()))
    evaluation_end = time.time()

    evaluation_time = evaluation_end - evaluation_start
    print("Total Model evaluation time: {}".format(str(timedelta(seconds=evaluation_time))))

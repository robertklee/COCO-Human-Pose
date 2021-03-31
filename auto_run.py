#%% 
import hourglass
import imp
imp.reload(hourglass)
from hourglass import HourglassNet
from constants import *
import matplotlib.pyplot as plt
import os

import data_generator
imp.reload(data_generator)
import data_generator
import time

import re
import pandas as pd
import evaluation
from evaluation import Evaluation
imp.reload(evaluation)
from HeatMap import HeatMap

#import HeatMap
#imp.reload(HeatMap)

import numpy as np

#%% find duplicates in list

def find_dup(pair_list, element):
    for (a,b) in pair_list:
        if element == a or element == b:
            return True
    return False

#%% 

def find_epochs(base_dir, sub_dir, epoch_dic, visited_models):
    directory = os.listdir(os.path.join(base_dir, sub_dir))
    weight_file = [f for f in directory if (f.endswith(".hdf5"))]
    for name in weight_file:
        match = re.search('hpe_epoch(\d+)', name)
        if match:
            epoch.append((match.group(1), sub_dir))

    resume_file = [f for f in os.listdir(base_dir) if (sub_dir in f and f is not sub_dir)]
    for file in resume_file:
        visited_models.append(file)
        for name in os.listdir(os.path.join(base_dir, file)):
            match = re.search('hpe_epoch(\d+)', name)
            #dup = find_dup(epoch, match.group(1))
            #print("match is ", match, "dup is ", dup)
            if match and not find_dup(epoch, match.group(1)):
                epoch.append((match.group(1), file))

    return list(set(epoch))


#%% save stacked evaluation heatmaps

def stacked_eval_heatmaps(test_df,eval):

    generator = data_generator.DataGenerator(
            df=test_df,
            base_dir=DEFAULT_VAL_IMG_PATH,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            num_hg_blocks=eval.num_hg_blocks,
            shuffle=False,  
            batch_size=len(test_df),
            online_fetch=False)

    # Select image to predict heatmaps
    X_batch, y_stacked = generator[0] # There is only one batch in the generator
    # X_batch, y_stacked = evaluation.load_and_preprocess_img('data/skier.jpg', eval.num_hg_blocks)
    y_batch = y_stacked[0] # take first hourglass section
    # Save stacked heatmap images to disk
    m_batch = test_df.to_dict('records') # TODO: eventually this will be passed from data generator as metadata
    print("\n\nEval start:   {}\n".format(time.ctime()))
    eval.visualize_batch(X_batch, y_batch, m_batch)
    print("\n\nEval end:   {}\n".format(time.ctime()))

#%% 

def predict_kp_bbox(val_df, eval):
    generator = data_generator.DataGenerator(
                df=val_df,
                base_dir=DEFAULT_VAL_IMG_PATH,
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                num_hg_blocks=DEFAULT_NUM_HG,
                shuffle=False,  
                batch_size=1,
                online_fetch=False)

    # Select image to predict heatmaps
    X_batch, y_stacked = generator[168] # choose one image for evaluation
    y_batch = y_stacked[0] # take first hourglass section
    X, y = X_batch[0], y_batch[0] # take first example of batch

    # Get predicted heatmaps for image
    predict_heatmaps=eval.predict_heatmaps(X_batch)

    # Get predicted keypoints from last hourglass (eval.num_hg_blocks-1)
    keypoints = eval.heatmaps_to_keypoints(predict_heatmaps[eval.num_hg_blocks-1, 0, :, :, :])
    print(keypoints)
    # Get bounding box image from heatmap
    heatmap = y[:,:,0]
    hm = HeatMap(X,heatmap)
    img = np.array(hm.image)

    # Clear plot image
    plt.clf()
    # Plot predicted keypoints on bounding box image
    x = []
    y = []
    for i in range(NUM_COCO_KEYPOINTS):
        if(keypoints[i,0] != 0 and keypoints[i,1] != 0):
            x.append(keypoints[i,0])
            y.append(keypoints[i,1])
    plt.scatter(x,y)
    plt.imshow(img)


# %% find each model's cooresponding epochs 

h = HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
_, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,0.1)

representative_set_df = pd.read_pickle(os.path.join(DEFAULT_PICKLE_PATH, 'representative_set.pkl'))
models = [] # keep track of the models that are visited 
epoch = [] # extract out all epoch numbers (across different files) from model logs
output_models = os.listdir(DEFAULT_OUTPUT_BASE_DIR) # keep track of the models have been explored before

for sub_dir in os.listdir(DEFAULT_MODEL_BASE_DIR):
    if '_hg_' in sub_dir and sub_dir not in models and sub_dir not in output_models:
        models.append(sub_dir)
        epoch = find_epochs(DEFAULT_MODEL_BASE_DIR, sub_dir, epoch, models)
        for (n_epoch, model_file) in epoch:
            eval = evaluation.Evaluation(
                    model_sub_dir=model_file,
                    epoch=int(n_epoch))
            stacked_eval_heatmaps(representative_set_df,eval)
            predict_kp_bbox(val_df,eval)

        


# %%

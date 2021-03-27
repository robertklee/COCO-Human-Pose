# %% Load Annotations into dataframes
import hourglass
import imp
imp.reload(hourglass)
from hourglass import HourglassNet
from constants import *
import matplotlib.pyplot as plt
import os

h = HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
_, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,0.1)

# %% Declare evaluation class instance
import evaluation
import HeatMap
imp.reload(evaluation)
imp.reload(HeatMap)

subdir = '2021-03-22-20h-23m_batchsize_12_hg_8_loss_weighted_mse_aug_medium_resume_2021-03-25-20h-02m'
eval = evaluation.Evaluation(
    model_sub_dir=subdir,
    epoch=43)

# %% Save stacked evaluation heatmaps
import data_generator
imp.reload(data_generator)
import data_generator
import time

generator = data_generator.DataGenerator(
            df=val_df,
            base_dir=DEFAULT_VAL_IMG_PATH,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            num_hg_blocks=eval.num_hg_blocks,
            shuffle=False,  
            batch_size=3,
            online_fetch=False)

# Select image to predict heatmaps
X_batch, y_stacked = generator[412] # choose one image for evaluation: 412 is tennis women
# X_batch, y_stacked = evaluation.load_and_preprocess_img('data/skier.jpg', eval.num_hg_blocks)
y_batch = y_stacked[0] # take first hourglass section
# Save stacked heatmap images to disk
m_batch = [{'img_id': 'first'}, {'img_id' : 'second'}, {'img_id' : 'third'}]
print("\n\nEval start:   {}\n".format(time.ctime()))
eval.visualize_batch(X_batch, y_batch, m_batch)
print("\n\nEval end:   {}\n".format(time.ctime()))

# %%

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
import pandas as pd
import evaluation
import HeatMap
imp.reload(evaluation)
imp.reload(HeatMap)

representative_set_df = pd.read_pickle(os.path.join(DEFAULT_PICKLE_PATH, 'representative_set.pkl'))
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
            df=representative_set_df,
            base_dir=DEFAULT_VAL_IMG_PATH,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            num_hg_blocks=eval.num_hg_blocks,
            shuffle=False,  
            batch_size=len(representative_set_df),
            online_fetch=False)

# Select image to predict heatmaps
X_batch, y_stacked = generator[0] # There is only one batch in the generator
# X_batch, y_stacked = evaluation.load_and_preprocess_img('data/skier.jpg', eval.num_hg_blocks)
y_batch = y_stacked[0] # take first hourglass section
# Save stacked heatmap images to disk
m_batch = representative_set_df.to_dict('records') # TODO: eventually this will be passed from data generator as metadata
print("\n\nEval start:   {}\n".format(time.ctime()))
eval.visualize_batch(X_batch, y_batch, m_batch)
print("\n\nEval end:   {}\n".format(time.ctime()))

# %% Plot predicted keypoints from heatmaps on bounding box image
import numpy as np
from HeatMap import HeatMap

generator = DataGenerator(
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
predict_heatmaps=eval.predict_heatmaps(h, X)

# Get predicted keypoints from last hourglass (eval.num_hg_blocks-1)
keypoints = eval.heatmaps_to_keypoints(predict_heatmaps[eval.num_hg_blocks-1, 0, :, :, :])

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
# %%

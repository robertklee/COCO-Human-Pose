# %% Load Annotations into dataframes
import hourglass
import imp
imp.reload(hourglass)
from hourglass import HourglassNet
from constants import *
import matplotlib.pyplot as plt
import os
# %matplotlib inline

#%%
h = HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
train_df, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,0.1)

# %% Declare evaluation class instance
import pandas as pd
import evaluation
import HeatMap
imp.reload(evaluation)
imp.reload(HeatMap)
# %%

# %%
representative_set_df = pd.read_pickle(os.path.join(DEFAULT_PICKLE_PATH, 'representative_set.pkl'))
<<<<<<< HEAD
subdir = '2021-04-01-21h-59m_batchsize_16_hg_4_loss_weighted_mse_aug_light_sigma4_learningrate_5.0e-03_opt_rmsProp_gt-4kp_activ_sigmoid_subset_0.50_lrfix'
eval = evaluation.Evaluation(
    model_sub_dir=subdir,
    epoch=26)
=======
subdir = '2021-03-28-09h-24m_batchsize_16_hg_4_loss_keras_mse_aug_light_sigma4_learningrate_5.0e-03_opt_adam_gt-4kp_activ_linear_subset_0.50_resume_2021-03-28-22h-10m'
eval = evaluation.Evaluation(
    model_sub_dir=subdir,
    epoch=30)
>>>>>>> 149070f... squash changes for rebase

# %% Save stacked evaluation heatmaps
import data_generator
imp.reload(data_generator)
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

<<<<<<< HEAD
generator = data_generator.DataGenerator(
            df=val_df,
            base_dir=DEFAULT_VAL_IMG_PATH,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            num_hg_blocks=DEFAULT_NUM_HG,
            shuffle=False,
            batch_size=1,
            online_fetch=False)

=======
>>>>>>> 149070f... squash changes for rebase
# Select image to predict heatmaps
X_batch, y_stacked = generator[168] # choose one image for evaluation
name_no_extension = "tmp"

## Uncomment below for arbitrary images
<<<<<<< HEAD
img_name = 'IMG_8175.jpg'
name_no_extension = img_name.split('.')[0]
X_batch, y_stacked = evaluation.load_and_preprocess_img(os.path.join('data', img_name), eval.num_hg_blocks)
=======
# img_name = 'IMG_3274.jpg'
# name_no_extension = img_name.split('.')[0]
# X_batch, y_stacked = evaluation.load_and_preprocess_img(os.path.join('data', img_name), eval.num_hg_blocks)
>>>>>>> 149070f... squash changes for rebase
y_batch = y_stacked[0] # take first hourglass section
X, y = X_batch[0], y_batch[0] # take first example of batch

# Get predicted heatmaps for image
predict_heatmaps=eval.predict_heatmaps(X_batch)

# Get predicted keypoints from last hourglass (last element of list)
# Dimensions are (hourglass_layer, batch, x, y, keypoint)
keypoints = eval.heatmaps_to_keypoints(predict_heatmaps[-1, 0, :, :, :])
print(keypoints)
# Get bounding box image from heatmap
heatmap = y[:,:,0]
hm = HeatMap(X,heatmap)
img = np.array(hm.image)

# Clear plot image
plt.clf()
<<<<<<< HEAD
eval.visualize_keypoints(np.zeros(INPUT_DIM), keypoints, name_no_extension + '_no-bg')
eval.visualize_keypoints(X, keypoints, name_no_extension)

=======
# Plot predicted keypoints on bounding box image
x = []
y = []
for i in range(NUM_COCO_KEYPOINTS):
    if(keypoints[i,0] != 0 and keypoints[i,1] != 0):
      x.append(keypoints[i,0])
      y.append(keypoints[i,1])
plt.scatter(x,y)
plt.imshow(img)

plt.savefig(os.path.join(DEFAULT_OUTPUT_BASE_DIR, f'{name_no_extension}_saved_scatter.png'))
>>>>>>> 149070f... squash changes for rebase
# %%

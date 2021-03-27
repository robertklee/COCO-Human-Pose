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
imp.reload(evaluation)
import evaluation

subdir = '2021-03-22-23h-55m_batchsize_12_hg_4_loss_weighted_mse_aug_medium_sigma4'

eval = evaluation.Evaluation(
    base_dir=DEFAULT_MODEL_BASE_DIR,
    sub_dir=subdir,
    epoch=58,
    h_net=h)
print("Created Evaluation instance")

# %% Save stacked evaluation heatmaps
import data_generator
imp.reload(data_generator)
import data_generator

generator = data_generator.DataGenerator(
            df=val_df,
            base_dir=DEFAULT_VAL_IMG_PATH,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            num_hg_blocks=eval.num_hg_blocks,
            shuffle=False,  
            batch_size=1,
            online_fetch=False)

# Select image to predict heatmaps
X_batch, y_stacked = generator[168] # choose one image for evaluation: 412 is tennis women
# X_batch, y_stacked = evaluation.load_and_preprocess_img('data/skier.jpg', eval.num_hg_blocks)
y_batch = y_stacked[0] # take first hourglass section
X, y = X_batch[0], y_batch[0] # take first example of batch
plt.imshow(X)
# Save stacked heatmap images to disk
filename = 'heatmap_evaluation.png'
eval.save_stacked_evaluation_heatmaps(X, y, os.path.join(DEFAULT_OUTPUT_BASE_DIR, filename))
print(f"Saved stacked evaluation heatmaps as {filename} to disk")

# %%

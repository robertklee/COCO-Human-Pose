# %% Load Annotations into dataframes
import hourglass
import imp
imp.reload(hourglass)
from hourglass import HourglassNet
import data_generator
imp.reload(data_generator)
from data_generator import DataGenerator
import evaluation
imp.reload(evaluation)
from evaluation import Evaluation
from constants import *
import matplotlib.pyplot as plt

h = HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
train_df, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,0.1)

# %% Declare evaluation class instance
eval = Evaluation(
    # ensure model_json and weights files exist in current directory and num_hg_blocks matches model_json
    model_json='hpe_hourglass_stacks_04_batchsize_012.json',
    weights='hpe_epoch71_val_loss_0.0417_train_loss_0.0163.hdf5',
    df=val_df,
    num_hg_blocks=DEFAULT_NUM_HG,
    batch_size=1)
print("Created Evaluation instance")

# %% Save stacked evaluation heatmaps
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
X_batch, y_stacked = generator[168] # choose one image for evaluation
y_batch = y_stacked[0] # take first hourglass section
X, y = X_batch[0], y_batch[0] # take first example of batch

# Save stacked heatmap images to disk
stacked_predict_heatmaps_file = 'stacked_predict_heatmaps.png'
stacked_ground_truth_heatmaps_file = 'stacked_ground_truth_heatmaps.png'
filename = 'heatmap_evaluation.png'
eval.save_stacked_evaluation_heatmaps(h, X, y, stacked_predict_heatmaps_file, stacked_ground_truth_heatmaps_file, filename)
print(f"Saved stacked evaluation heatmaps as {filename} to disk")

# %%

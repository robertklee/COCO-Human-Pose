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

# Retrieve prediction and ground truth heatmaps
predict_heatmaps=eval.predict_heatmaps(h, X)
print("Received predicted heatmaps")
stacked_predict_heatmaps=eval.stacked_predict_heatmaps(predict_heatmaps)
print("Received stacked predicted heatmaps")
stacked_ground_truth_heatmaps=eval.stacked_ground_truth_heatmaps(X, y)
print("Received stacked ground truth heatmaps")

# Save stacked images to disk
plt.imsave('stacked_predict_heatmaps.png', stacked_predict_heatmaps)
plt.imsave('stacked_ground_truth_heatmaps.png', stacked_ground_truth_heatmaps)
filename = 'heatmap_evaluation.png'
eval.save_stacked_evaluation_heatmaps('stacked_predict_heatmaps.png', 'stacked_ground_truth_heatmaps.png', filename)
print(f"Saved stacked evaluation heatmaps as {filename}")

# %%

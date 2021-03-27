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
    model_json='models/hpe_hourglass_stacks_04_batchsize_012.json',
    weights='models/hpe_epoch71_val_loss_0.0417_train_loss_0.0163.hdf5',
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
plt.plot(x,y)
plt.imshow(img)
# %%

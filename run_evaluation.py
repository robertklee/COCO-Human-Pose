# %% Load Annotations into dataframes
import hourglass
import imp
imp.reload(hourglass)
from hourglass import HourglassNet
from constants import *
import matplotlib.pyplot as plt

h = HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
_, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,0.1)

# %% Declare evaluation class instance
import evaluation
import HeatMap
imp.reload(evaluation)
imp.reload(HeatMap)
import evaluation

eval = evaluation.Evaluation(
    model_json='hpe_hourglass_stacks_08_batchsize_012.json',
    weights='hpe_epoch36_val_loss_415393.8125_train_loss_0.0334.hdf5',
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
            num_hg_blocks=DEFAULT_NUM_HG,
            shuffle=False,  
            batch_size=1,
            online_fetch=False)

# Select image to predict heatmaps
X_batch, y_stacked = generator[168] # choose one image for evaluation: 412 is tennis women
y_batch = y_stacked[0] # take first hourglass section
X, y = X_batch[0], y_batch[0] # take first example of batch
plt.imshow(X)
# Save stacked heatmap images to disk
filename = 'heatmap_evaluation.png'
eval.save_stacked_evaluation_heatmaps(X, y, filename)
print(f"Saved stacked evaluation heatmaps as {filename} to disk")

# %%

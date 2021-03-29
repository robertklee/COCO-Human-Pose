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
# %%import hourglass
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
import numpy as np

h = HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
train_df, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,0.1)

batch_size = 1

eval = Evaluation(
    # ensure model_json and weights files exist in current directory and num_hg_blocks matches model_json
    model_json='hpe_hourglass_stacks_08_batchsize_012.json',
    weights='hpe_epoch34_val_loss_0.1417_train_loss_0.1577.hdf5',
    df=val_df,
    num_hg_blocks=DEFAULT_NUM_HG,
    batch_size=batch_size)
print("Created Evaluation instance")

generator = DataGenerator(
            df=val_df,
            base_dir=DEFAULT_TRAIN_IMG_PATH,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            num_hg_blocks=DEFAULT_NUM_HG,
            shuffle=False,
            batch_size=batch_size,
            online_fetch=True,
            is_eval=True)

# Select image to predict heatmaps
# X_batch, y_stacked = generator[168] # choose one image for evaluation
# y_batch = y_stacked[0] # take first hourglass section
# X, y = X_batch[0], y_batch[0] # take first example of batch

# Select image to predict heatmaps
X_batch, y_stacked, metadatas = generator[168] # choose one image for evaluation
y_batch = y_stacked[0] # take first hourglass section
X, y = X_batch[0], y_batch[0] # take first example of batch

untransformed_predictions = [(28, 25, 0.060447669346224),
 (29, 22, 0.060447669346224),
 (27, 22, 0.06401293421084475),
 (35, 21, 0.06778848201428475),
 (0, 0, 0.0),
 (46, 34, 0.06401293421084475),
 (28, 27, 0.06401293421084474),
 (0, 0, 0.0),
 (22, 34, 0.06778848201428475),
 (37, 47, 0.06401293421084475),
 (16, 36, 0.06401293421084474),
 (36, 55, 0.06402403246820193),
 (24, 51, 0.060447669346224),
 (0, 0, 0.0),
 (0, 0, 0.0),
 (0, 0, 0.0),
 (0, 0, 0.0)]

untransformed_predictions = np.array(untransformed_predictions).flatten()
print(untransformed_predictions)
metadata = metadatas[0]
print(metadata)
metadata = eval.undo_bounding_box_transformations(metadata, untransformed_predictions)
print(metadata['predicted_labels'])



# %%

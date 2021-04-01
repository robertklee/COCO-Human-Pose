# %% Load Annotations into dataframes
import matplotlib
# %matplotlib notebook
# %matplotlib inline
matplotlib.use('TkAgg')
import hourglass
import imp
imp.reload(hourglass)
from hourglass import HourglassNet
from constants import *
import numpy as np

h = HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
train_df, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,0.1)

# %% Declare function  to display ground truth images
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pylab
import coco_df
from matplotlib.patches import Rectangle
pylab.rcParams['figure.figsize'] = (30.0, 30.0)
import cv2

train_annot_path = DEFAULT_TRAIN_ANNOT_PATH
val_annot_path = DEFAULT_VAL_ANNOT_PATH
# train_coco = COCO(train_annot_path) # load annotations for training set
# val_coco = COCO(val_annot_path) # load annotations for validation set

# df = coco_df.get_df(train_annot_path,val_annot_path)

# %% Test the generator
import data_generator
imp.reload(data_generator)
from data_generator import DataGenerator
from constants import *

generator = DataGenerator(
    df=val_df,
    base_dir=DEFAULT_TRAIN_IMG_PATH,
    input_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM,
    num_hg_blocks=DEFAULT_NUM_HG,
    shuffle=False,
    batch_size=1,
    online_fetch=False,
    img_aug_strength=ImageAugmentationStrength.light)

# Test the generator
import time
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

from submodules.HeatMap import HeatMap # https://github.com/LinShanify/HeatMap

start = time.time()
X_batch, y_stacked, kps_batch = generator[238]
print("Retrieving batch took: ",time.time() - start, " s")
y_batch = y_stacked[0] # take first hourglass section
print("Batch shape is: ", X_batch.shape, y_batch.shape)
X, y = X_batch[0], y_batch[0] # take first example of batch
kps = kps_batch[0]
print("Example shape is: ", X.shape,y.shape)

# %% Show one example
# Transpose so the dimensions are (keypoint, x,y)
heatmaps = np.transpose(y, axes=(2,0,1))
implot = plt.imshow(X)

ptx = [kps[i*NUM_COCO_KP_ATTRBS] for i in range(NUM_COCO_KEYPOINTS)]
pty = [kps[i*NUM_COCO_KP_ATTRBS+1] for i in range(NUM_COCO_KEYPOINTS)]

pt_joined = np.vstack((ptx, pty))

pt_filtered = pt_joined[:, np.all(pt_joined > 0, axis=0)]

plt.scatter(pt_filtered[0], pt_filtered[1])
plt.show()

# %% show all keypoints
fig = plt.figure()
plt.imshow(np.hstack([heatmaps[i] for i in range(NUM_COCO_KEYPOINTS)]))

# fig2 = plt.figure()
# plt.imshow(X)

summed_heatmaps_mono = np.sum(heatmaps, axis=0)
summed_heatmaps = np.stack((summed_heatmaps_mono,)*3, axis=-1)
rescaled_summed_heatmaps = cv2.resize(summed_heatmaps, INPUT_DIM, interpolation = cv2.INTER_AREA)

# fig3 = plt.figure()
# plt.imshow(summed_heatmaps)

blended_img = cv2.addWeighted(X, 0.7, rescaled_summed_heatmaps, 0.5, 0)
fig4 = plt.figure()
plt.imshow(blended_img)

plt.show()

# %%

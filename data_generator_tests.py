# %% Load Annotations into dataframes
import hourglass
import imp
imp.reload(hourglass)
from hourglass import HourglassNet
from constants import *

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

train_annot_path = DEFAULT_TRAIN_ANNOT_PATH
val_annot_path = DEFAULT_VAL_ANNOT_PATH
train_coco = COCO(train_annot_path) # load annotations for training set
val_coco = COCO(val_annot_path) # load annotations for validation set

df = coco_df.get_df(train_annot_path,val_annot_path)

def display_img(annId):
  # Determine if img exists and if it is in train or val set
  img_df_rows = df.loc[df['ann_id'] == annId]
  if len(img_df_rows) == 0:
      print(f"Image with ann id {annId} does not exist.")
      return

  coco = train_coco if img_df_rows['source'].iloc[0] == 0 else val_coco

  # Get img id from file name
  imgId = img_df_rows['src_set_image_id'].iloc[0]
  img = coco.imgs[imgId]
  I = io.imread(img['coco_url']) # load image from URL (no need to store image locally)

  # load and display keypoints annotations
  plt.subplot(1,2,1)
  plt.imshow(I)
  plt.axis('off')

  plt.subplot(1,2,2)
  plt.imshow(I)
  annIds = coco.getAnnIds(imgIds=[imgId])
  anns = coco.loadAnns(annIds)
  bbox= list(img_df_rows['bbox'])[0]
  plt.gca().add_patch(Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],
                  edgecolor='red',
                  facecolor='none',
                  lw=4))
  coco.showAnns(anns)
  plt.show()

# %% Test the generator
import pandas as pd
import data_augmentation
imp.reload(data_augmentation)
import data_generator
imp.reload(data_generator)
from data_generator import DataGenerator


representative_set_df = pd.read_pickle(os.path.join(DEFAULT_PICKLE_PATH, 'representative_set.pkl'))
generator = DataGenerator(
    df=representative_set_df,
    base_dir=DEFAULT_VAL_IMG_PATH,
    input_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM,
    num_hg_blocks=DEFAULT_NUM_HG,
    shuffle=False,
    batch_size=1,
    online_fetch=False,
    img_aug_strength=ImageAugmentationStrength.heavy)

# Test the generator
import time
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

from HeatMap import HeatMap # https://github.com/LinShanify/HeatMap

start = time.time()
X_batch, y_stacked = generator[0]
print("Retrieving batch took: ",time.time() - start, " s")
y_batch = y_stacked[0] # take first hourglass section
print("Batch shape is: ", X_batch.shape, y_batch.shape)
X, y = X_batch[0], y_batch[0] # take first example of batch
print("Example shape is: ", X.shape,y.shape)

for i in range(NUM_COCO_KEYPOINTS):
    print(COCO_KEYPOINT_LABEL_ARR[i])
    heatmap = y[:,:,i]
    hm = HeatMap(X,heatmap)
    hm.plot(transparency=0.5,show_axis=True,show_colorbar=True)
# %%

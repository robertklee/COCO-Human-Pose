
from hourglass import HourglassNet
from data_generator import DataGenerator
from constants import *
import time

batch_size = 15

h = HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,3,INPUT_DIM,OUTPUT_DIM)
train_df, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH)

generator = DataGenerator(
    df=val_df,
    base_dir=DEFAULT_TRAIN_IMG_PATH,
    input_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM,
    num_hg_blocks=DEFAULT_NUM_HG,
    shuffle=True,
    batch_size=batch_size,
    online_fetch=True
)

print(f"Generator has {len(generator)} batches, each of size {batch_size}")
start = time.time()
for X, y in generator:
    end = time.time()
    print(f"Retrieving batch took {end-start} seconds")
    start = time.time()
import os
from enum import Enum

import numpy as np

# Auto-detect Google Colab environment
try:
    import google.colab
    COLAB_TRAINING = True
except ImportError:
    COLAB_TRAINING = os.path.exists('/content')

# Model Defaults
DEFAULT_NUM_HG = 4 # NOTE in their final design, 8 were used
DEFAULT_EPOCHS = 70
DEFAULT_DATA_BASE_DIR = 'data'
DEFAULT_MODEL_BASE_DIR = 'models'
DEFAULT_LOGS_BASE_DIR = 'logs'
DEFAULT_OUTPUT_BASE_DIR = 'output'

DEFAULT_RESUME_DIR_FLAG = '_resume_'
HPE_EPOCH_PREFIX = 'hpe_epoch'
HPE_HOURGLASS_STACKS_PREFIX = 'hpe_hourglass_stacks'

# Loss Functions
class LossFunctionOptions(Enum):
    keras_mse = 0
    weighted_mse = 1
    euclidean_loss = 2
    focal_loss = 3

DEFAULT_LOSS = LossFunctionOptions.keras_mse.name

# Image augmentation
class ImageAugmentationStrength(Enum):
    none = 0
    light = 1
    medium = 2
    heavy = 3
DEFAULT_AUGMENT = ImageAugmentationStrength.none.name

# Output activation
class OutputActivation(Enum):
    linear = 0
    sigmoid = 1
    relu = 2
DEFAULT_ACTIVATION = OutputActivation.linear.name

# Optimizer
class OptimizerType(Enum):
    rmsProp = 0
    adam = 1
DEFAULT_OPTIMIZER = OptimizerType.adam.name

DEFAULT_LEARNING_RATE = 0.005

# Dataset Constants
if COLAB_TRAINING:
    # DEFAULT_TRAIN_ANNOT_PATH = '/content/gdrive/My Drive/COCO/2017/annotations/person_keypoints_train2017.json'
    # DEFAULT_VAL_ANNOT_PATH = '/content/gdrive/My Drive/COCO/2017/annotations/person_keypoints_val2017.json'
    DEFAULT_TRAIN_ANNOT_PATH = '/content/datasets/annotations/person_keypoints_train2017.json'
    DEFAULT_VAL_ANNOT_PATH = '/content/datasets/annotations/person_keypoints_val2017.json'
    DEFAULT_TRAIN_IMG_PATH = '/content/datasets/coco'
    DEFAULT_VAL_IMG_PATH = '/content/datasets/coco'
    DEFAULT_PICKLE_PATH = '/content/gdrive/MyDrive/Colab Data/COCO-Human-Pose/Pickles'
else:
    DEFAULT_TRAIN_ANNOT_PATH = 'data/annotations/person_keypoints_train2017.json'
    DEFAULT_VAL_ANNOT_PATH = 'data/annotations/person_keypoints_val2017.json'
    DEFAULT_TRAIN_IMG_PATH = 'data/coco'
    DEFAULT_VAL_IMG_PATH = 'data/coco'
    DEFAULT_PICKLE_PATH = 'Pickles'

# Backcompatability constants.
COCO_TRAIN_ANNOT_PATH = DEFAULT_TRAIN_ANNOT_PATH
COCO_VAL_ANNOT_PATH = DEFAULT_VAL_ANNOT_PATH

# Order of keypoints in COCO dataset
COCO_KEYPOINT_LABEL_ARR = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
                           "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

# This array was copied from https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/dataset/coco.py
# The original array was 1-indexed, so we subtract 1 from each element
COCO_SKELETON = np.array([
                    [16, 14],
                    [14, 12],
                    [17, 15],
                    [15, 13],
                    [12, 13],
                    [6, 12],
                    [7, 13],
                    [6, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10],
                    [9, 11],
                    [2, 3],
                    [1, 2],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7]]) - 1

# Colouring for linking joints together (legacy, kept for backward compatibility)
COLOUR_MAP = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            #   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#B73CC7', '#0258C9', '#f76f8e', '#048ba8', '#16db93',
              '#f06449', '#0f7173', '#276FBF', '#791e94', '#407899',
              '#e71d36', '#FFA69E', '#678EC2', '#069E7D']

# Body-region skeleton bone colors (RGB tuples), indexed to match COCO_SKELETON.
# Left side = warm tones (red/orange), right side = cool tones (blue/cyan),
# torso = green, head = yellow.
SKELETON_BONE_COLORS = [
    (255, 100, 100),  # left_ankle  → left_knee   (left leg)
    (255, 130, 80),   # left_knee   → left_hip    (left leg)
    (100, 100, 255),  # right_ankle → right_knee  (right leg)
    (80, 130, 255),   # right_knee  → right_hip   (right leg)
    (100, 210, 100),  # left_hip    → right_hip   (torso)
    (100, 210, 100),  # left_shoulder → left_hip   (torso)
    (100, 210, 100),  # right_shoulder → right_hip (torso)
    (100, 210, 100),  # left_shoulder → right_shoulder (torso)
    (255, 160, 60),   # left_shoulder → left_elbow (left arm)
    (60, 160, 255),   # right_shoulder → right_elbow (right arm)
    (255, 180, 40),   # left_elbow  → left_wrist  (left arm)
    (40, 180, 255),   # right_elbow → right_wrist  (right arm)
    (230, 230, 80),   # left_eye    → right_eye   (head)
    (230, 230, 80),   # nose        → left_eye    (head)
    (230, 230, 80),   # nose        → right_eye   (head)
    (220, 210, 90),   # left_eye    → left_ear    (head)
    (220, 210, 90),   # right_eye   → right_ear   (head)
    (180, 210, 120),  # left_ear    → left_shoulder  (head/torso)
    (180, 210, 120),  # right_ear   → right_shoulder (head/torso)
]

# Per-keypoint colors (RGB tuples), indexed by COCO keypoint index.
# Follows the same warm-left / cool-right convention.
KEYPOINT_COLORS = [
    (255, 255, 255),  # 0:  nose           (center - white)
    (230, 220, 100),  # 1:  left_eye       (head warm)
    (100, 220, 230),  # 2:  right_eye      (head cool)
    (230, 200, 80),   # 3:  left_ear       (head warm)
    (80, 200, 230),   # 4:  right_ear      (head cool)
    (255, 160, 60),   # 5:  left_shoulder  (left arm)
    (60, 160, 255),   # 6:  right_shoulder (right arm)
    (255, 140, 50),   # 7:  left_elbow     (left arm)
    (50, 140, 255),   # 8:  right_elbow    (right arm)
    (255, 180, 40),   # 9:  left_wrist     (left arm)
    (40, 180, 255),   # 10: right_wrist    (right arm)
    (255, 100, 100),  # 11: left_hip       (left leg)
    (100, 100, 255),  # 12: right_hip      (right leg)
    (255, 80, 80),    # 13: left_knee      (left leg)
    (80, 80, 255),    # 14: right_knee     (right leg)
    (255, 60, 60),    # 15: left_ankle     (left leg)
    (60, 60, 255),    # 16: right_ankle    (right leg)
]

# Model parameters
INPUT_DIM = (256,256)
INPUT_CHANNELS = 3
OUTPUT_DIM = (64,64)
NUM_CHANNELS = 256

# Data Generator Constants
DEFAULT_BATCH_SIZE = 12 #NOTE need to test optimal batch size
NUM_COCO_KEYPOINTS = 17 # Number of joints to detect
NUM_COCO_KP_ATTRBS = 3 # (x,y,v) * 17 keypoints
BBOX_SLACK = 1.3 # before augmentation, increase bbox size to 130%

KP_FILTERING_GT = 4 # Greater than x keypoints

# NOTE: The effective sigma is downscaled by a factor of 4 (from (256,256) to (64,64)) on each side, so ensure the sigma is appropriately sized
HEATMAP_SIGMA = 4 # As per https://towardsdatascience.com/human-pose-estimation-with-stacked-hourglass-network-and-tensorflow-c4e9f84fd3ce
REVERSE_HEATMAP_SIGMA = 1 # Use reverse sigma when heatmap is reversed by a factor of 4 from (64,64) to (256,256)

# NOTE: Don't use, the scale is applied in the loss function.
HEATMAP_SCALE = 1 #TODO figure out if we want to scale heatmaps, set to 1 for no effect. There are 82 times background pixels to foreground pixels in 7*7 patch of 64*64 heatmap, see same link for HEATMAP_SIGMA
TRAIN_SHUFFLE = True # Can set to false for debug purposes
VAL_SHUFFLE = False

# Data filtering constants
BBOX_MIN_SIZE = 900 # Filter out images smaller than 30x30, TODO tweak

# Output filtering constants
HM_TO_KP_THRESHOLD = 0.2
HM_TO_KP_THRESHOLD_POST_FILTER = HM_TO_KP_THRESHOLD / 5

# Weighted centroid parameters for keypoint coordinate refinement
# Number of brightness std devs below peak to include in the centroid region
WEIGHTED_CENTROID_BRIGHTNESS_K = 2
# Number of spatial std devs from peak to include in the centroid region
WEIGHTED_CENTROID_SPATIAL_K = 2

PCK_THRESHOLD = 0.2
# This default PCK threshold is used when either hip is not present.
# It is empirically chosen by taking the mean torso width in the validation set
# {'mean': 25.273791558055517, 'std': 19.27466898776274}
DEFAULT_PCK_THRESHOLD = PCK_THRESHOLD * 25

# Output save names
OUTPUT_STACKED_HEATMAP = 'heatmaps'
OUTPUT_SKELETON = 'skeleton'
OUTPUT_USER_IMG = 'user_img'

class Generator(Enum):
    train_gen = 0
    val_gen = 1
    representative_set_gen = 2

class Metrics(Enum):
    pck = 0
    oks = 1

# Data aug flip R/L probability
RL_FLIP = 0.5

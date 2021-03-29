from enum import Enum

# Colab Training
COLAB_TRAINING = False

# Model Defaults
DEFAULT_NUM_HG = 4 # NOTE in their final design, 8 were used
DEFAULT_EPOCHS = 40
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
    DEFAULT_PICKLE_PATH = '/content/gdrive/MyDrive/Colab Data/COCO-Human-Pose/Pickles/'
else:
    DEFAULT_TRAIN_ANNOT_PATH = 'data/annotations/person_keypoints_train2017.json'
    DEFAULT_VAL_ANNOT_PATH = 'data/annotations/person_keypoints_val2017.json'
    DEFAULT_TRAIN_IMG_PATH = 'data/coco'
    DEFAULT_VAL_IMG_PATH = 'data/coco'
    DEFAULT_PICKLE_PATH = './Pickles/'

COCO_KEYPOINT_LABEL_ARR = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

# Backcompatability constants. 
COCO_TRAIN_ANNOT_PATH = DEFAULT_TRAIN_ANNOT_PATH
COCO_VAL_ANNOT_PATH = DEFAULT_VAL_ANNOT_PATH

# Model parameters
INPUT_DIM = (256,256)
INPUT_CHANNELS = 3
OUTPUT_DIM = (64,64)
NUM_CHANNELS = 256

# Data Generator Constants
DEFAULT_BATCH_SIZE = 10 #NOTE need to test optimal batch size
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

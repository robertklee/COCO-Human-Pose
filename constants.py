# Colab Training
COLAB_TRAINING = False

# Model Defaults
DEFAULT_NUM_HG = 4 # NOTE in their final design, 8 were used
DEFAULT_EPOCHS = 40
DEFAULT_MODEL_PATH = 'models'
DEFAULT_LOGS_PATH = 'logs'
DEFAULT_OUTPUT_PATH = 'output'

# Dataset Constants
if COLAB_TRAINING:
    DEFAULT_TRAIN_ANNOT_PATH = '/content/gdrive/My Drive/COCO/2017/annotations/person_keypoints_train2017.json'
    DEFAULT_VAL_ANNOT_PATH = '/content/gdrive/My Drive/COCO/2017/annotations/person_keypoints_val2017.json'
    DEFAULT_TRAIN_IMG_PATH = '/content/datasets/coco'
    DEFAULT_VAL_IMG_PATH = '/content/datasets/coco'
else:
    DEFAULT_TRAIN_ANNOT_PATH = 'data/annotations/person_keypoints_train2017.json'
    DEFAULT_VAL_ANNOT_PATH = 'data/annotations/person_keypoints_val2017.json'
    DEFAULT_TRAIN_IMG_PATH = 'data/coco'
    DEFAULT_VAL_IMG_PATH = 'data/coco'

COCO_TRAIN_ANNOT_PATH = DEFAULT_TRAIN_ANNOT_PATH
COCO_VAL_ANNOT_PATH = DEFAULT_VAL_ANNOT_PATH

# Model parameters
INPUT_DIM = (256,256)
OUTPUT_DIM = (64,64)
NUM_CHANNELS = 256

# Data Generator Constants
DEFAULT_BATCH_SIZE = 10 #NOTE need to test optimal batch size
NUM_COCO_KEYPOINTS = 17 # Number of joints to detect
NUM_COCO_KP_ATTRBS = 3 # (x,y,v) * 17 keypoints

TRAIN_SHUFFLE = True # Can set to false for debug purposes
VAL_SHUFFLE = False

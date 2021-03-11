# Number of joints to detect
num_classes = 16 #TODO change to 17 after verifying model
num_hg_default = 4 # NOTE in their final design, 8 were used
INPUT_DIM = (256,256)
OUTPUT_DIM = (64,64)

# Data Generator Constants
DEFAULT_BATCH_SIZE = 10 #NOTE need to test optimal batch size
NUM_COCO_KEYPOINTS = 17
NUM_COCO_KP_ATTRBS = 3 # (x,y,v) * 17 keypoints

# COCO DF Constants
COCO_TRAIN_ANNOT_PATH = '/content/gdrive/My Drive/COCO/2017/annotations/person_keypoints_train2017.json'
COCO_VAL_ANNOT_PATH = '/content/gdrive/My Drive/COCO/2017/annotations/person_keypoints_val2017.json'
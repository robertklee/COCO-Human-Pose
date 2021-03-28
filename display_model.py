 %%
import csv
from hourglass import HourglassNet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from constants import *
from data_generator import DataGenerator

import os
import numpy as np
import scipy.misc
#from mpii_datagen import MPIIDataGen
#from eval_heatmap import get_predicted_kp_from_htmap
import argparse
#from pckh import run_pckh

# %%
def main_eval(model_json, model_weights, num_stack, num_class, matfile, tiny, batch_size):
    #inres = (192, 192) if tiny else (256, 256)
    #outres = (48, 48) if tiny else (64, 64)

    inres = (256, 256)
    outres = (64, 64)

    xnet = HourglassNet(num_classes=NUM_COCO_KEYPOINTS, num_stacks=num_stack, num_channels=NUM_CHANNELS, inres=INPUT_DIM,
                            outres=OUTPUT_DIM)

    xnet._load_model(model_json, model_weights)

    train_df, val_df = xnet.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,0.1)

    val_data = DataGenerator(val_df, DEFAULT_VAL_IMG_PATH, inres, outres, num_stack, shuffle=VAL_SHUFFLE, batch_size=1)
    
    print('val data size', len(val_df))

    valkps = np.zeros(shape=(len(val_df), NUM_COCO_KEYPOINTS, 2), dtype=np.float)
    count = 0
    
    
    for X_batch, y_stacked  in val_data:
        out = xnet.model.predict(X_batch)
        #print(out)
        with open('out.txt', 'w') as f:
            print(out, file = f)

        #get_final_pred_kps(valkps, out[-1], _meta, outres)

    #scipy.io.savemat(matfile, mdict={'preds': out})

    #run_pckh(model_json, matfile)

    #X_batch, y_stacked  = val_data[0]
 
    #out = xnet.model.predict(X_batch)

    #print(out)

        #get_final_pred_kps(valkps, out[-1], _meta, outres)

    #scipy.io.savemat(matfile, mdict={'preds': out})

    #run_pckh(model_json, matfile)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ##parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--model_json", help='path to store trained model')
    parser.add_argument("--model_weights", help='path to store trained model')
    parser.add_argument("--mat_file", help='path to store trained model')
    parser.add_argument("--num_stack", type=int, help='num of stack')
    parser.add_argument("--tiny", default=False, type=bool, help="tiny network for speed, inres=[192x128], channel=128")
    parser.add_argument("--batch",type=int, default=DEFAULT_BATCH_SIZE, help='batch size')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    main_eval(model_json=args.model_json, model_weights=args.model_weights, matfile=args.mat_file,
              num_stack=args.num_stack, num_class=NUM_COCO_KEYPOINTS, tiny=args.tiny, batch_size = args.batch)


# %%

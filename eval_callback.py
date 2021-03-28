import keras
import os
import datetime
from time import time
from data_generator import DataGenerator
from constants import *
from hourglass import HourglassNet
# from eval_heatmap import cal_heatmap_acc TODO: figure out where meta is stored 


class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath, inres, outres):
        self.foldpath = foldpath
        self.inres = inres
        self.outres = outres

    def get_folder_path(self):
        return self.foldpath

    def run_eval(self, epoch):
        xnet = HourglassNet(num_classes=NUM_COCO_KEYPOINTS, num_stacks=num_stack, num_channels=NUM_CHANNELS, inres=INPUT_DIM,
                            outres=OUTPUT_DIM)
        train_df, val_df = xnet.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH, subset)
        valdata = DataGenerator(val_df, DEFAULT_VAL_IMG_PATH, self.inres, self.outres, self.num_stacks, shuffle=VAL_SHUFFLE, batch_size=batch_size, img_aug_strength=None)

        total_suc, total_fail = 0, 0
        threshold = 0.5

        count = 0

        # TODO: meta data info needed 
        for X_batch, y_stacked  in valdata:

            count += batch_size
            if count > (len(val_df):
                break

            out = self.model.predict(X_batch)

            #suc, bad = cal_heatmap_acc(out[-1], _meta, threshold) # TODO

            total_suc += suc
            total_fail += bad

        acc = total_suc * 1.0 / (total_fail + total_suc)

        print('Eval Accuray ', acc, '@ Epoch ', epoch)

        with open(os.path.join(self.get_folder_path(), 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + '\n')

    def on_epoch_end(self, epoch, logs=None):
        # This is a walkaround to sovle model.save() issue
        # in which large network can't be saved due to size.

        # save model to json
        if epoch == 0:
            jsonfile = os.path.join(self.foldpath, "net_arch.json")
            with open(jsonfile, 'w') as f:
                f.write(self.model.to_json())

        # save weights
        modelName = os.path.join(self.foldpath, "weights_epoch" + str(epoch) + ".h5")
        self.model.save_weights(modelName)

        print("Saving model to ", modelName)

        self.run_eval(epoch)
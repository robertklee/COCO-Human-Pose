from datetime import datetime
import os

import numpy as np
import scipy.misc
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.losses import mean_squared_error
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop

from hourglass_blocks import (bottleneck_block, bottleneck_mobile,
                              create_hourglass_network)
from data_generator import DataGenerator
import coco_df
from constants import *

# Some code adapted from https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/blob/master/src/net/hourglass.py

class HourglassNet(object):

    def __init__(self, num_classes, num_stacks, num_channels, inres, outres):
        self.num_classes = num_classes
        self.num_stacks = num_stacks
        self.num_channels = num_channels
        self.inres = inres
        self.outres = outres

    def build_model(self, mobile=False, show=False):
        if mobile:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                  self.num_channels, self.inres, self.outres, bottleneck_mobile)
        else:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                  self.num_channels, self.inres, self.outres, bottleneck_block)
        # show model summary and layer name
        if show:
            self.model.summary()
    
    def load_and_filter_annotations(self,path_to_train_anns,path_to_val_anns):
        df = coco_df.get_df(path_to_train_anns,path_to_val_anns)
        # apply filters here
        print(f"Unfiltered df contains {len(df)} anns")
        df = df.loc[df['is_crowd'] == 0] # filter crowd
        df = df.loc[df['bbox_area'] > 900] # filter small bboxes (30*30)
        print(f"Filtered df contains {len(df)} anns")
        train_df = df.loc[df['source'] == 0]
        val_df = df.loc[df['source'] == 1]
        return train_df.reset_index(), val_df.reset_index()

    def train(self, batch_size, model_path, epochs):
        current_time = datetime.today().strftime('%Y-%m-%d-%Hh-%Mm')

        train_df, val_df = self.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH)
        #train_df, val_df = load_and_filter_annotations(COCO_TRAIN_ANNOT_PATH,COCO_VAL_ANNOT_PATH) # for google collab NOTE no longer needed
        train_generator = DataGenerator(train_df, DEFAULT_TRAIN_IMG_PATH, self.inres, self.outres, self.num_stacks, shuffle=TRAIN_SHUFFLE, batch_size=batch_size)
        val_generator = DataGenerator(val_df, DEFAULT_VAL_IMG_PATH, self.inres, self.outres, self.num_stacks, shuffle=VAL_SHUFFLE, batch_size=batch_size)
        
        csv_logger = CSVLogger(os.path.join(model_path, 'csv_tr' + current_time + '.csv'))

        modelSavePath = os.path.join(model_path, current_time +  '_batchsize_' + str(batch_size), 'hg_epoch{epoch:02d}_val_loss_{val_loss:.4f}_train_loss_{loss:.4f}.hdf5')

        mc_val = ModelCheckpoint(modelSavePath, monitor='val_loss')
        mc_train = ModelCheckpoint(modelSavePath, monitor='loss')
        tb = TensorBoard(log_dir=os.path.join(DEFAULT_LOGS_PATH, current_time + '_batchsize_' + str(batch_size)), histogram_freq=0, write_graph=True, write_images=True)

        # TODO potentially add learning rate scheduler callback

        callbacks = [mc_val, mc_train, tb, csv_logger]

        print("Model saved to: {}".format(modelSavePath))

        self.model.fit_generator(generator=train_generator, validation_data=val_generator, steps_per_epoch=len(train_generator), \
            validation_steps=len(val_generator), epochs=epochs, callbacks=callbacks)
    
    # TODO resume and load model
    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):
        # TODO
        pass

    def load_model(self, modeljson, modelfile):
        # TODO
        pass
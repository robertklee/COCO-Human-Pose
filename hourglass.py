from datetime import datetime
import os

import numpy as np
import scipy.misc
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.losses import mean_squared_error
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
import tensorflow as tf

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
    
    def load_and_filter_annotations(self, path_to_train_anns,path_to_val_anns, subset):
        df = coco_df.get_df(path_to_train_anns,path_to_val_anns)
        # apply filters here
        print(f"Unfiltered df contains {len(df)} anns")
        df = df.loc[df['is_crowd'] == 0] # drop crowd anns
        df = df.loc[df['num_keypoints'] != 0] # drop anns containing no kps
        df = df.loc[df['bbox_area'] > BBOX_MIN_SIZE] # drop small bboxes
        train_df = df.loc[df['source'] == 0]
        val_df = df.loc[df['source'] == 1]
        if subset != 1.0:
            train_df = train_df.sample(frac=subset)
        print(f"Train/Val dfs contains {len(train_df)}/{len(val_df)} anns")
        return train_df.reset_index(), val_df.reset_index()

    def _start_train(self, batch_size, model_base_dir, epochs, initial_epoch, model_subdir, current_time, subset):
        self._compile_model()

        train_df, val_df = self.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH, DEFAULT_VAL_ANNOT_PATH, subset)

        train_generator = DataGenerator(train_df, DEFAULT_TRAIN_IMG_PATH, self.inres, self.outres, self.num_stacks, shuffle=TRAIN_SHUFFLE, batch_size=batch_size)
        val_generator = DataGenerator(val_df, DEFAULT_VAL_IMG_PATH, self.inres, self.outres, self.num_stacks, shuffle=VAL_SHUFFLE, batch_size=batch_size)
        

        modelDir = os.path.join(model_base_dir, model_subdir)
        logsDir = os.path.join(DEFAULT_LOGS_BASE_DIR, model_subdir)

        modelSavePath = os.path.join(modelDir, '{prefix}{{epoch:02d}}_val_loss_{{val_loss:.4f}}_train_loss_{{loss:.4f}}.hdf5'.format(prefix=HPE_EPOCH_PREFIX))

        if not os.path.exists(modelDir):
            print("Model save directory created: {}".format(modelDir))
            os.makedirs(modelDir)

        # Create callbacks
        mc_val = ModelCheckpoint(modelSavePath, monitor='val_loss')
        mc_train = ModelCheckpoint(modelSavePath, monitor='loss')
        csv_logger = CSVLogger(os.path.join(modelDir, 'csv_tr' + current_time + '.csv'))
        tb = TensorBoard(log_dir=logsDir, histogram_freq=0, write_graph=True, write_images=True)

        # TODO potentially add learning rate scheduler callback

        callbacks = [mc_val, mc_train, tb, csv_logger]

        architecture_json_file = os.path.join(modelDir, '{}_{:02d}_batchsize_{:03d}.json'.format(HPE_HOURGLASS_STACKS_PREFIX, self.num_stacks, batch_size))
        if not os.path.exists(architecture_json_file):
            with open(architecture_json_file, 'w') as f:
                print("Model architecture json saved to: {}".format(architecture_json_file))
                f.write(self.model.to_json())

        print("Model checkpoints saved to: {}".format(modelSavePath))

        self.model.fit(train_generator, validation_data=val_generator, steps_per_epoch=len(train_generator), \
            validation_steps=len(val_generator), epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks)

    def train(self, batch_size, model_save_base_dir, epochs, subset):
        current_time = datetime.today().strftime('%Y-%m-%d-%Hh-%Mm')

        model_subdir = current_time + '_batchsize_' + str(batch_size) + '_hg_' + str(self.num_stacks)

        self._start_train(batch_size=batch_size, model_base_dir=model_save_base_dir, epochs=epochs, initial_epoch=0, model_subdir=model_subdir, current_time=current_time, subset=subset)
    
    def resume_train(self, batch_size, model_save_base_dir, model_json, model_weights, init_epoch, epochs, resume_subdir, subset):
        if resume_subdir is not None:
            print('Automatically locating model architecture .json and weights .hdf5...')

        print('Restoring model architecture json: {}'.format(model_json))
        print('Restoring model weights: {}'.format(model_weights))

        self._load_model(model_json, model_weights)

        current_time = datetime.today().strftime('%Y-%m-%d-%Hh-%Mm')

        # for consistency, identify original model subdirectory and create a new subdir of the same name, suffixed by a resume time flag
        orig_model_subdir = os.path.basename(os.path.dirname(model_weights))

        # If resuming a previously resumed training session (to prevent _resume_ from appended over and over)
        if DEFAULT_RESUME_DIR_FLAG in orig_model_subdir:
            # strip everything after resume flag
            orig_model_subdir = orig_model_subdir[:orig_model_subdir.find(DEFAULT_RESUME_DIR_FLAG)]

        model_subdir = orig_model_subdir + DEFAULT_RESUME_DIR_FLAG + current_time

        self._start_train(batch_size=batch_size, model_base_dir=model_save_base_dir, epochs=epochs, initial_epoch=init_epoch, model_subdir=model_subdir, current_time=current_time, subset=subset)
    
    def _compile_model(self):
        tf.keras.backend.clear_session()

        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))

        strategy = tf.distribute.TPUStrategy(resolver)

        with strategy.scope():
            model = create_hourglass_network(self.num_classes, self.num_stacks, self.num_channels, self.inres, self.outres, bottleneck_block)
            model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=5e-4), loss=mean_squared_error, metrics=["accuracy"])
            # TODO Update optimizer and/or learning rate?
        self.model = model

    def _load_model(self, model_json, model_weights):
        with open(model_json) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(model_weights)
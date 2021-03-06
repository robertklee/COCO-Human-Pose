import datetime
import os

import numpy as np
import scipy.misc
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.losses import mean_squared_error
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop

from hourglass_blocks import (bottleneck_block, bottleneck_mobile,
                              create_hourglass_network)


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
        
    def train(self, batch_size, model_path, epochs):
        # TODO
        pass
    
    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):
        # TODO
        pass

    def load_model(self, modeljson, modelfile):
        # TODO
        pass
import os

import pandas as pd
from pycocotools.coco import COCO

import data_generator
import evaluation
import hourglass
from constants import *


class EvaluationWrapper():

    def __init__(self, model_sub_dir):
        self.epoch = 24 # TODO: write method to determine default epoch
        self.eval = evaluation.Evaluation(model_sub_dir=model_sub_dir, epoch=self.epoch)
        representative_set_df = pd.read_pickle(os.path.join(DEFAULT_PICKLE_PATH, 'representative_set.pkl'))
        self.representative_set_gen = data_generator.DataGenerator( df=representative_set_df,
                                                                    base_dir=DEFAULT_VAL_IMG_PATH,
                                                                    input_dim=INPUT_DIM,
                                                                    output_dim=OUTPUT_DIM,
                                                                    num_hg_blocks=1, # does not matter
                                                                    shuffle=False,
                                                                    batch_size=len(representative_set_df), # single batch
                                                                    online_fetch=False,
                                                                    is_eval=True)
        h = hourglass.HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
        _, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,1.0)
        self.val_gen = data_generator.DataGenerator(df=val_df,
                                                    base_dir=DEFAULT_VAL_IMG_PATH,
                                                    input_dim=INPUT_DIM,
                                                    output_dim=OUTPUT_DIM,
                                                    num_hg_blocks=1, # does not matter
                                                    shuffle=False,
                                                    batch_size=DEFAULT_BATCH_SIZE,
                                                    online_fetch=False,
                                                    is_eval=True)
        self.cocoGt = COCO(DEFAULT_VAL_ANNOT_PATH)
        print("Initialized Evaluation Wrapper!")

    def visualizeHeatmaps(self, images='representative_set'):
        pass

    def visalizeKeypoints(self, images='representative_set'):
        pass

    def calculateOKS(self, epochs):
        image_ids, list_of_predictions = self._full_list_of_predicitons(self.representative_set_gen)
        oks = self.eval.oks_eval(image_ids, list_of_predictions, self.cocoGt)
        return oks

    def calculatePCK(self, epochs):
        _, list_of_predictions = self._full_list_of_predicitons(self.representative_set_gen)
        pck = self.eval.pck_eval(list_of_predictions)
        avg = sum(pck.values())
        print(avg/len(pck))
        return pck

    def plotOKS(self, epochs):
        pass

    def plotPCK(self, epochs):
        pass

    def _full_list_of_predicitons(self, gen):
        list_of_predictions = []
        image_ids = []
        for X_batch, _, metadata_batch in gen:
            predicted_heatmaps_batch = self.eval.predict_heatmaps(X_batch)
            imgs, predictions = self.eval.heatmap_to_COCO_format(predicted_heatmaps_batch, metadata_batch)
            list_of_predictions += predictions
            image_ids += imgs
        return image_ids, list_of_predictions

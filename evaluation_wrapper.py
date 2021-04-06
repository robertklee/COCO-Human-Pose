import os
from functools import lru_cache

import pandas as pd
from pycocotools.coco import COCO

import data_generator
import evaluation
import hourglass
import util
from constants import *


class EvaluationWrapper():

    def __init__(self, model_sub_dir, epoch=None, model_base_dir=DEFAULT_MODEL_BASE_DIR):
        self.model_sub_dir = model_sub_dir

        if epoch is None:
            self.epoch = util.get_highest_epoch_file(model_base_dir=model_base_dir, model_subdir=model_sub_dir)

            print(f'Automatically using largest epoch {self.epoch:3d}...\n')
        else:
            self.epoch = epoch

        self.eval = evaluation.Evaluation(model_base_dir=model_base_dir, model_sub_dir=model_sub_dir, epoch=self.epoch)
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

    def visualizeHeatmaps(self, genEnum=Generator.representative_set_gen):
        self.visualize(genEnum=genEnum, visualize_heatmaps=True, visualize_scatter=False, visualize_skeleton=False)

    def visualizeKeypoints(self, genEnum=Generator.representative_set_gen, visualize_skeleton=True):
        self.visualize(genEnum=genEnum, visualize_heatmaps=False, visualize_scatter=True, visualize_skeleton=visualize_skeleton)

    # Heatmaps is by default False because it is extremely processor intensive to calculate
    def visualize(self, genEnum=Generator.representative_set_gen, visualize_heatmaps=False, visualize_scatter=True, visualize_skeleton=True):
        gen = self._get_generator(genEnum)

        gen_length = len(gen)
        for i in range(gen_length):
            X_batch, y_stacked, m_batch = gen[i]

            y_batch = y_stacked[0] # take first hourglass section

            img_id_batch = [m['ann_id'] for m in m_batch] # image IDs are the annotation IDs

            predicted_heatmaps_batch = self.eval.predict_heatmaps(X_batch)

            if visualize_heatmaps:
                self.eval.visualize_heatmaps(X_batch, y_batch, img_id_batch, predicted_heatmaps_batch)

            if visualize_scatter or visualize_skeleton:
                # Get predicted keypoints from last hourglass (last element of list)
                # Dimensions are (hourglass_layer, batch, x, y, keypoint)
                keypoints_batch = self.eval.heatmaps_to_keypoints_batch(predicted_heatmaps_batch)

                if visualize_skeleton:
                    # Plot only skeleton
                    self.eval.visualize_keypoints(np.zeros(X_batch.shape), keypoints_batch, [img_id + '_no_bg' for img_id in img_id_batch])

                # Plot skeleton with image
                self.eval.visualize_keypoints(X_batch, keypoints_batch, img_id_batch, show_skeleton=visualize_skeleton)

            util.print_progress_bar(1.0*i/gen_length, label=f"Batch {i}/{gen_length}")

        # Flush any leftover progress bar to 100%
        util.print_progress_bar(1, label=f"Batch {gen_length}/{gen_length}")
        print()

    def calculateOKS(self, epochs, genEnum):
        gen = self._get_generator(genEnum)
        image_ids, list_of_predictions = self._full_list_of_predictions(gen, self.model_sub_dir, self.epoch)
        oks = self.eval.oks_eval(image_ids, list_of_predictions, self.cocoGt)
        print(oks)
        return oks

    def calculatePCK(self, epochs, genEnum):
        gen = self._get_generator(genEnum)
        _, list_of_predictions = self._full_list_of_predictions(gen, self.model_sub_dir, self.epoch)
        pck = self.eval.pck_eval(list_of_predictions)
        return pck

    def plotOKS(self, epochs):
        pass

    def plotPCK(self, epochs):
        pass

    @lru_cache(maxsize=50)
    def _full_list_of_predictions(self, gen, model_sub_dir, epoch):
        list_of_predictions = []
        image_ids = []

        gen_length = len(gen)
        for i in range(gen_length):
            X_batch, _, metadata_batch = gen[i]

            predicted_heatmaps_batch = self.eval.predict_heatmaps(X_batch)
            imgs, predictions = self.eval.heatmap_to_COCO_format(predicted_heatmaps_batch, metadata_batch)
            list_of_predictions += predictions
            image_ids += imgs

            util.print_progress_bar(1.0*i/gen_length, label=f"Batch {i}/{gen_length}")

        # Flush any leftover progress bar to 100%
        util.print_progress_bar(1, label=f"Batch {gen_length}/{gen_length}")
        print()

        return image_ids, list_of_predictions

    def _get_generator(self, genEnum):
        if genEnum == Generator.representative_set_gen:
            gen = self.representative_set_gen
        elif genEnum == Generator.val_gen:
            gen = self.val_gen

        return gen

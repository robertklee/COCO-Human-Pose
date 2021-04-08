import csv
import os
from functools import lru_cache

import keras
import pandas as pd
from pycocotools.coco import COCO

import data_generator
import evaluation
import hourglass
import util
from constants import *


class EvaluationWrapper():

    def __init__(self, model_sub_dir, epoch=None, model_base_dir=DEFAULT_MODEL_BASE_DIR):
        self.update_model(model_sub_dir, epoch=None, model_base_dir=DEFAULT_MODEL_BASE_DIR)

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


    """
    K.clear_session() is useful when you're creating multiple models in succession,
    such as during hyperparameter search or cross-validation. Each model you train
    adds nodes (potentially numbering in the thousands) to the graph. TensorFlow
    executes the entire graph whenever you (or Keras) call tf.Session.run() or
    tf.Tensor.eval(), so your models will become slower and slower to train, and you
    may also run out of memory. Clearing the session removes all the nodes left over
    from previous models, freeing memory and preventing slowdown.

    See https://stackoverflow.com/questions/50895110/what-do-i-need-k-clear-session-and-del-model-for-keras-with-tensorflow-gpu
    """
    def __del__(self):
        # Clear backend session to prevent running out of memory
        keras.backend.clear_session()

    def update_model(self, model_sub_dir, epoch=None, model_base_dir=DEFAULT_MODEL_BASE_DIR):
        # Clear backend session to prevent running out of memory
        keras.backend.clear_session()

        self.model_sub_dir = model_sub_dir

        if epoch is None:
            self.epoch = util.get_highest_epoch_file(model_base_dir=model_base_dir, model_subdir=model_sub_dir)

            print(f'Automatically using largest epoch {self.epoch:3d}...\n')
        else:
            self.epoch = epoch

        self.eval = evaluation.Evaluation(model_base_dir=model_base_dir, model_sub_dir=model_sub_dir, epoch=self.epoch)

    def predict_on_image(self, img_path, img_id, visualize_skeleton=True):
        # Number of hg blocks doesn't matter
        X_batch, y_stacked = evaluation.load_and_preprocess_img(img_path, 1)

        img_id_batch = [img_id]

        # y_batch = y_stacked[0] # take first hourglass section
        # X, y = X_batch[0], y_batch[0] # take first example of batch

        # Get predicted heatmaps for image
        predicted_heatmaps_batch = self.eval.predict_heatmaps(X_batch)

        # Get predicted keypoints from last hourglass (last element of list)
        # Dimensions are (hourglass_layer, batch, x, y, keypoint)
        keypoints_batch = self.eval.heatmaps_to_keypoints_batch(predicted_heatmaps_batch)

        # Get predicted keypoints from last hourglass (last element of list)
        # Dimensions are (hourglass_layer, batch, x, y, keypoint)
        keypoints_batch = self.eval.heatmaps_to_keypoints_batch(predicted_heatmaps_batch)

        if visualize_skeleton:
            # Plot only skeleton
            img_id_batch_bg = [f'{img_id}_no_bg' for img_id in img_id_batch]
            self.eval.visualize_keypoints(np.zeros(X_batch.shape), keypoints_batch, img_id_batch_bg, show_skeleton=visualize_skeleton)

        # Plot skeleton with image
        self.eval.visualize_keypoints(X_batch, keypoints_batch, img_id_batch, show_skeleton=visualize_skeleton)

    def visualizeHeatmaps(self, genEnum=Generator.representative_set_gen):
        self.visualize(genEnum=genEnum, visualize_heatmaps=True, visualize_scatter=False, visualize_skeleton=False)

    def visualizeKeypoints(self, genEnum=Generator.representative_set_gen, visualize_skeleton=True, average_flip_prediction=True):
        self.visualize(genEnum=genEnum, visualize_heatmaps=False, visualize_scatter=True, visualize_skeleton=visualize_skeleton, average_flip_prediction=average_flip_prediction)

    # Heatmaps is by default False because it is extremely processor intensive to calculate
    def visualize(self, genEnum=Generator.representative_set_gen, visualize_heatmaps=False, visualize_scatter=True, visualize_skeleton=True, average_flip_prediction=True):
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
                    img_id_batch_bg = [f'{img_id}_no_bg' for img_id in img_id_batch]
                    self.eval.visualize_keypoints(np.zeros(X_batch.shape), keypoints_batch, img_id_batch_bg, show_skeleton=visualize_skeleton)

                # Plot skeleton with image
                self.eval.visualize_keypoints(X_batch, keypoints_batch, img_id_batch, show_skeleton=visualize_skeleton)

            util.print_progress_bar(1.0*i/gen_length, label=f"Batch {i}/{gen_length}")

        # Flush any leftover progress bar to 100%
        util.print_progress_bar(1, label=f"Batch {gen_length}/{gen_length}")
        print()

    def calculateOKS(self, epochs, genEnum, average_flip_prediction=False):
        gen = self._get_generator(genEnum)
        image_ids, list_of_predictions = self._full_list_of_predictions_wrapper(gen, self.model_sub_dir, self.epoch, average_flip_prediction=average_flip_prediction)
        oks = self.eval.oks_eval(image_ids, list_of_predictions, self.cocoGt)
        self._append_to_results_file('oks.csv', oks, oks.keys())


    def calculatePCK(self, epochs, genEnum, average_flip_prediction=False):
        # one epoch provided
        # or no epoch provided
        # or range of epochs provided
        gen = self._get_generator(genEnum)
        _, list_of_predictions = self._full_list_of_predictions_wrapper(gen, self.model_sub_dir, self.epoch, average_flip_prediction=average_flip_prediction)
        pck = self.eval.pck_eval(list_of_predictions)
        self._append_to_results_file('pck.csv', pck, pck.keys())

    def _append_to_results_file(self, file_name, row_dict, header):
        row_dict['epoch'] = self.epoch
        row_dict['model'] = self.model_sub_dir
        file_path = os.path.join(DEFAULT_OUTPUT_BASE_DIR, self.model_sub_dir, file_name)
        results_exist = os.path.isfile(file_path)
        with open(file_path, 'a') as f:
            dict_writer = csv.DictWriter(f, delimiter=',', fieldnames=header)
            if not results_exist:
                dict_writer.writeheader()
            dict_writer.writerow(row_dict)

    def plotOKS(self, epochs):
        pass

    def plotPCK(self, epochs):
        pass

    def _full_list_of_predictions_wrapper(self, gen, model_sub_dir, epoch, average_flip_prediction=False):
        print('Predicting over all batches...')
        image_ids, list_of_predictions = self._full_list_of_predictions(gen, self.model_sub_dir, self.epoch, predict_using_flip=False)
        print()

        if average_flip_prediction:
            print('Predicting over all batches using a horizontally flipped input, with prediction coordinates transformed back...')
            image_ids_2, list_of_predictions_2 = self._full_list_of_predictions(gen, self.model_sub_dir, self.epoch, predict_using_flip=True)
            print()

            assert image_ids == image_ids_2, "Expected the image IDs should be in the same order"
            eps = 1e-3
            for i in range(len(list_of_predictions)):
                # Average predictions from original image and the untransformed flipped image to get a more accurate prediction
                for j in range(NUM_COCO_KEYPOINTS):
                    # This code is required so if one version detects the keypoint (x,y,1),
                    # and the other doesn't (0,0,0), we don't average them to be (x/2, y/2, 0.5)
                    base = j * NUM_COCO_KP_ATTRBS

                    n = 0
                    x_sum = 0
                    y_sum = 0

                    if list_of_predictions[i]['keypoints'][base+2] == 1:
                        x_sum += list_of_predictions[i]['keypoints'][base]
                        y_sum += list_of_predictions[i]['keypoints'][base + 1]
                        n += 1

                    if list_of_predictions_2[i]['keypoints'][base+2] == 1:
                        x_sum += list_of_predictions_2[i]['keypoints'][base]
                        y_sum += list_of_predictions_2[i]['keypoints'][base + 1]
                        n += 1

                    if n > 0:
                        list_of_predictions[i]['keypoints'][base] = round(x_sum / n)
                        list_of_predictions[i]['keypoints'][base + 1] = round(y_sum / n)
                        list_of_predictions[i]['keypoints'][base + 2] = 1

                    # list_of_predictions[i]['keypoints'] = np.round(np.mean( np.array([ list_of_predictions[i]['keypoints'], list_of_predictions_2[i]['keypoints'] ]), axis=0 ))

        return image_ids, list_of_predictions

    @lru_cache(maxsize=50)
    def _full_list_of_predictions(self, gen, model_sub_dir, epoch, predict_using_flip=False):
        list_of_predictions = []
        image_ids = []

        gen_length = len(gen)
        for i in range(gen_length):
            X_batch, _, metadata_batch = gen[i]

            # X_batch has dimensions (batch, x, y, channels)
            # Run both original and flipped image through and average the predictions
            # Typically increases accuracy by a few percent
            if predict_using_flip:
                # Copy by reference NOTE X_batch is modified __in place__
                # Horizontal flip each image in batch
                X_batch_flipped = X_batch[:,:,::-1,:]

                # Feed flipped image into model
                # output shape is (num_hg_blocks, X_batch_size, 64, 64, 17)
                predicted_heatmaps_batch_flipped = self.eval.predict_heatmaps(X_batch_flipped)

                # indices to flip order of Left and Right heatmaps [0, 2, 1, 4, 3, 6, 5, 8, 7, etc]
                reverse_LR_indices = [0] + [2*x-y for x in range(1,9) for y in range(2)]

                # reverse horizontal flip AND reverse left/right heatmaps
                predicted_heatmaps_batch = predicted_heatmaps_batch_flipped[:,:,:,::-1,reverse_LR_indices]
            else:
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

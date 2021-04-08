# %% Create Eval Instance
import os
import time
from datetime import datetime, timedelta

import imp
import evaluation_wrapper
imp.reload(evaluation_wrapper)
import evaluation
imp.reload(evaluation)
import constants
import matplotlib.pyplot as plt
%matplotlib inline

eval = evaluation_wrapper.EvaluationWrapper('2021-04-03-18h-11m_batchsize_16_hg_4_loss_keras_mse_aug_medium_sigma4_learningrate_5.0e-03_opt_adam_gt-4kp_activ_sigmoid')
# eval = evaluation_wrapper.EvaluationWrapper('2021-04-01-21h-59m_batchsize_16_hg_4_loss_weighted_mse_aug_light_sigma4_learningrate_5.0e-03_opt_rmsProp_gt-4kp_activ_sigmoid_subset_0.50_lrfix')

# %% Run OKS
start = time.time()

eval.calculateOKS(1, constants.Generator.representative_set_gen, average_flip_prediction=True)
elapsed = time.time() - start
print("Total OKS average normal & flip time: {}".format(str(timedelta(seconds=elapsed))))

start = time.time()

eval.calculateOKS(1, constants.Generator.representative_set_gen, average_flip_prediction=False)

elapsed = time.time() - start
print("Total OKS time: {}".format(str(timedelta(seconds=elapsed))))

# %% Run PCK
start = time.time()

eval.calculatePCK(1, constants.Generator.representative_set_gen, average_flip_prediction=True)
eval.calculatePCK(1, constants.Generator.representative_set_gen, average_flip_prediction=False)

elapsed = time.time() - start
print("Total PCK time: {}".format(str(timedelta(seconds=elapsed))))

# %% Visualize representative batch heatmaps
start = time.time()

eval.visualizeHeatmaps(constants.Generator.representative_set_gen)

elapsed = time.time() - start
print("Total heatmap time: {}".format(str(timedelta(seconds=elapsed))))

# %% Visualize representative batch keypoints
start = time.time()

eval.visualizeKeypoints(constants.Generator.representative_set_gen)

elapsed = time.time() - start
print("Total keypoint time: {}".format(str(timedelta(seconds=elapsed))))
# %% Run on arbitrary images
img_name = 'IMG_3320.JPG'
name_no_extension = img_name.split('.')[0]
user_img_dir = 'user_img'

img_path = os.path.join(constants.DEFAULT_DATA_BASE_DIR, user_img_dir, img_name)

eval.predict_on_image(img_path, name_no_extension)
eval.predict_on_image(img_path, name_no_extension + '_flip', average_flip_prediction=True)
# %%

# %% Prepare evaluator and generator
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import re
import imp

import evaluation
import data_generator
imp.reload(evaluation)
imp.reload(data_generator)
from constants import *

representative_set_df = pd.read_pickle(os.path.join(DEFAULT_PICKLE_PATH, 'representative_set.pkl'))
subdir = '2021-03-31-08h-54m_batchsize_16_hg_4_loss_weighted_mse_aug_light_sigma4_learningrate_5.0e-03_opt_rmsProp_gt-4kp_activ_sigmoid_subset_0.50_wmse-1-5'

generator = data_generator.DataGenerator(
            df=representative_set_df,
            base_dir=DEFAULT_VAL_IMG_PATH,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            num_hg_blocks=1, # doesn't matter for evaluation b/c we take one stack for GT
            shuffle=False,
            batch_size=len(representative_set_df),
            online_fetch=False)

# %% Run visualization on epoch range and save images to disk

epochs_to_visualize = [27, 28] #range(34,45)
print("\n\nEval start:   {}\n".format(time.ctime()))
for epoch in epochs_to_visualize:
    eval = evaluation.Evaluation(
        model_sub_dir=subdir,
        epoch=epoch)
    X_batch, y_stacked = generator[0] # There is only one batch in the generator
    y_batch = y_stacked[0] # take first hourglass section
    m_batch = representative_set_df.to_dict('records') # TODO: eventually this will be passed from data generator as metadata
    eval.visualize_batch(X_batch, y_batch, m_batch)
print("\n\nEval end:   {}\n".format(time.ctime()))

# %%

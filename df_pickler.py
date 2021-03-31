# Pickles and unpickles data frames for maintaining consistant dataframes.

import numpy as np
import pandas as pd
import os
import skimage.io as io
import coco_df
from constants import *
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split




def check_pickle_folder(pickle_folder, pickle_path = DEFAULT_PICKLE_PATH) -> bool:
    # Checks if the folder exists
    return os.path.isdir(os.path.join(pickle_path, pickle_folder))


def create_folder(pickle_folder, pickle_path = DEFAULT_PICKLE_PATH):
    # Creates a folder
    pf = os.path.join(pickle_path, pickle_folder)
    if check_pickle_folder(pickle_folder, pickle_path):
        print(f'Folder {pickle_folder} exists')
        return pf
    print(pf)
    os.mkdir(pf)
    return pf

def make_pickle(is_crowd: bool, drop_empty: bool, scale: float,TV_split: float, drop_small_bbox=BBOX_MIN_SIZE):
    if scale <= 0.0 or scale > 1.0:
        raise ValueError
    if TV_split <= 0.0 or scale > 1.0:
        raise ValueError
    pickle_name = pickle_namer(is_crowd, drop_empty, scale, TV_split, drop_small_bbox)
    pickle_folder = create_folder(pickle_name)
    df = coco_df.get_df(DEFAULT_TRAIN_ANNOT_PATH, DEFAULT_VAL_ANNOT_PATH)
    if is_crowd:
        df = df.loc[df['is_crowd'] == 0]
    if drop_empty:
        df = df.loc[df['num_keypoints'] != 0]
    df = df.loc[df['bbox_area'] > drop_small_bbox]
    df_TrV = df.loc[df['source'] == 0] #Data from the training set for splitting into train/val
    df_Tst = df.loc[df['source'] == 1] #Data from the val set for testing
    t_size = scale * TV_split
    v_size = scale * (1-TV_split)
    df_Tr, df_V = train_test_split(df_TrV, train_size=t_size, test_size=v_size) # We can stratify here if wanted as well
    df_Tr.to_pickle(os.path.join(pickle_folder, f'Train_{pickle_name}.pkl'))
    df_V.to_pickle(os.path.join(pickle_folder, f'Val_{pickle_name}.pkl'))
    df_Tst.to_pickle(os.path.join(pickle_folder, f'Test_{pickle_name}.pkl'))
    return


def get_pickle(pickle_name):
    pickle_folder = os.path.join(DEFAULT_PICKLE_PATH, pickle_name)
    df_Tr = pd.read_pickle(os.path.join(pickle_folder, f'Train_{pickle_name}.pkl'))
    df_V = pd.read_pickle(os.path.join(pickle_folder, f'Val_{pickle_name}.pkl'))
    df_Tst = pd.read_pickle(os.path.join(pickle_folder, f'Test_{pickle_name}.pkl'))
    return df_Tr.reset_index(), df_V.reset_index(), df_Tst.reset_index()

def pickle_namer(is_crowd, drop_empty, scale, TV_split, drop_small_bbox):
    filename = f''
    filename += f'C-{is_crowd}_'
    filename += f'DE-{drop_empty}_'
    filename += f'S-{int(scale*100)}_'
    filename += f'TVs-{int(TV_split*100)}_'
    filename += f'DSBB-{drop_small_bbox}'
    return filename

def parse_pickle_name(filename):
    data =  [val for val in filename.split('.')[0].split('_')]
    print(f'Settings for pickle ({filename}):')
    print(f'  Crowd enabled              : {data[0].split("-")[1]}')
    print(f'  Drop Empty                 : {data[1].split("-")[1]}')
    print(f'  Dataframe scaling          : {data[2].split("-")[1]/100}')
    print(f'  Test/validation splot      : {data[3].split("-")[1]/100}')
    print(f'  Bounding box lower limit   : {data[4].split("-")[1]}')
    return

# %% Import required libraries
# Import utilities
from imgaug.augmenters.meta import OneOf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# Import Image manipulation
from PIL import Image

# Import data visualization
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib

# import the library and helpers
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

from constants import ImageAugmentationStrength

# Holy resources: https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B01%20-%20Augment%20Keypoints.ipynb
# Credit to the above notebook for their tutorial on keypoint augmentation

def _print_options():
    print('Image augmentation strength was not found in possible options.')
    print('Available options are:')
    available_strengths = [name for name, member in ImageAugmentationStrength.__members__.items()]
    print(available_strengths)
    exit(1)

def get_strength_enum_from_string(img_aug_strength_str):
    try:
        strength = ImageAugmentationStrength[img_aug_strength_str]
    except KeyError:
        _print_options()
    
    return strength

def get_pipeline(strength):
    if strength is ImageAugmentationStrength.none:
        return None
    elif strength is ImageAugmentationStrength.light:
        return light_augmentation()
    elif strength is ImageAugmentationStrength.medium:
        return medium_augmentation()
    elif strength is ImageAugmentationStrength.heavy:
        return heavy_augmentation()
    else:
        _print_options()

# Not applied transformations but would be interesting to try:
# - iaa.CoarseDropout - Randomly erases a larger chunk of the image - meant to improve robustness for occlusions
# - iaa.SaltAndPepper - Different type of noise

def light_augmentation():
    aug_pipeline = iaa.Sequential([
        iaa.Sometimes(0.3, iaa.GaussianBlur((0, 1.0))), # apply Gaussian blur with a sigma between 0 and 2 to 30% of the images # used to be [0,3] on 50% images
        iaa.Sometimes(0.5, iaa.Fliplr(1.0)), # horizontally flip 50% of the time
        # apply from 0 to 3 of the augmentations from the list
        iaa.SomeOf((0, 2),[
            iaa.Dropout((0, 0.03), per_channel=0.5), # randomly remove up to 3% of the pixels
            iaa.AddToHueAndSaturation((-15, 15)),  # change their color
            iaa.OneOf([
                iaa.AddToBrightness((-20,20)),
                # Strengthen or weaken the contrast in each image.
                iaa.LinearContrast((0.8, 1.2)),
            ]),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5),
        ]),
        # Only apply one of the following because otherwise there is a risk that keypoints will 
        # be pointing to a non-existent part of the image
        iaa.SomeOf((0,1),[
            iaa.CropAndPad(percent=(-0.15, 0.15), keep_size=True, sample_independently=False), # crop and pad 50% of the images
            iaa.Rotate((-15,15)), # rotate between [-30, 30] degrees
        ])
    ],
    random_order=True # apply the augmentations in random order
    )

    ## We may need this line so our pipeline will apply the same operations to keypoints as well as images
    ## Disabled because we will just pass both the keypoints and the image to the seq pipeline
    ## which eliminates the possibility that different augmentations will be applied on the 
    ## original image and keypoints 
    # aug_pipeline_det = aug_pipeline.to_deterministic()

    return aug_pipeline

def medium_augmentation():
    aug_pipeline = iaa.Sequential([
        iaa.Sometimes(0.3, iaa.GaussianBlur((0, 2.0))), # apply Gaussian blur with a sigma between 0 and 2 to 30% of the images # used to be [0,3] on 50% images
        iaa.Sometimes(0.5, iaa.Fliplr(1.0)), # horizontally flip 50% of the time
        # apply from 0 to 3 of the augmentations from the list
        iaa.SomeOf((0, 3),[
            iaa.Dropout((0, 0.05), per_channel=0.5), # randomly remove up to 5% of the pixels # used to be 10%
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            iaa.AddToHueAndSaturation((-20, 20)),  # change their color #(-60, 60)
            iaa.OneOf([
                iaa.AddToBrightness((-25,25)),
                # Strengthen or weaken the contrast in each image.
                iaa.LinearContrast((0.75, 1.25)),
            ]),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        ]),
        # Only apply one of the following because otherwise there is a risk that keypoints will 
        # be pointing to a non-existent part of the image
        iaa.SomeOf((0,1),[
            iaa.CropAndPad(percent=(-0.20, 0.20), keep_size=True, sample_independently=False), # crop and pad 50% of the images
            iaa.Rotate((-25,25)), # rotate 50% of the images between [-30, 30] degrees
        ])
    ],
    random_order=True # apply the augmentations in random order
    )

    ## We may need this line so our pipeline will apply the same operations to keypoints as well as images
    ## Disabled because we will just pass both the keypoints and the image to the seq pipeline
    ## which eliminates the possibility that different augmentations will be applied on the 
    ## original image and keypoints 
    # aug_pipeline_det = aug_pipeline.to_deterministic()

    return aug_pipeline

# %% init augmentation pipeline
# NOTE: DO NOT APPLY AFFINE SCALE AFTER CROP
# Keypoints that should have disappeared will be part of the image again:
# https://github.com/aleju/imgaug/issues/187
def heavy_augmentation():
    # Perform data augmentation randomly
            # - Rotation (+/- 30 deg)
            # - Scaling (.75 to 1.25)
            # - Horizontal flip (left to right) with probability 50%
            # - Gaussian noise 
            # - Random brightness
            # - Random gamma (contrast)
            # - Random dropout (fine dropout. Coarse may drop out the area where our keypoint is)

    # define an augmentation pipeline
    aug_pipeline = iaa.Sequential([
        iaa.Sometimes(0.3, iaa.GaussianBlur((0, 2.0))), # apply Gaussian blur with a sigma between 0 and 2 to 30% of the images # used to be [0,3] on 50% images
        iaa.Sometimes(0.5, iaa.Fliplr(1.0)), # horizontally flip 50% of the time
        # apply from 0 to 3 of the augmentations from the list
        iaa.SomeOf((0, 3),[
            iaa.Dropout((0, 0.1), per_channel=0.5), # randomly remove up to 5% of the pixels # used to be 10%
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            iaa.AddToHueAndSaturation((-30, 30)),  # change their color #(-60, 60)
            iaa.OneOf([
                iaa.AddToBrightness((-30,30)),
                # Strengthen or weaken the contrast in each image.
                iaa.LinearContrast((0.75, 1.5)),
            ]),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        ]),
        # Only apply one of the following because otherwise there is a risk that keypoints will 
        # be pointing to a non-existent part of the image
        iaa.SomeOf((0,1),[
            iaa.CropAndPad(percent=(-0.25, 0.25), keep_size=True, sample_independently=False), # crop and pad 50% of the images
            iaa.Rotate((-30,30)), # rotate 50% of the images between [-30, 30] degrees
        ])
    ],
    random_order=True # apply the augmentations in random order
    )

    ## We may need this line so our pipeline will apply the same operations to keypoints as well as images
    ## Disabled because we will just pass both the keypoints and the image to the seq pipeline
    ## which eliminates the possibility that different augmentations will be applied on the 
    ## original image and keypoints 
    # aug_pipeline_det = aug_pipeline.to_deterministic()

    return aug_pipeline

# %% Load sample image
if __name__ == '__main__':
    image = imageio.imread('./data/Macropus_rufogriseus_rufogriseus_Bruny.jpg')
    image = ia.imresize_single_image(image, (389, 259))

    ymax = 389
    xmax = 259
    offset = 4

    kps = [
        Keypoint(x=99, y=81),   # left eye (from camera perspective)
        Keypoint(x=125, y=80),  # right eye
        Keypoint(x=112, y=102), # nose
        Keypoint(x=102, y=210), # left paw
        Keypoint(x=127, y=207), # right paw
        # ***** Extrema to test out of bounds keypoints *****
        Keypoint(x=offset, y=offset),
        Keypoint(x=xmax-offset, y=offset),
        Keypoint(x=offset, y=ymax-offset),
        Keypoint(x=xmax-offset, y=ymax-offset),
        Keypoint(x=xmax*0.5, y=ymax-offset),
        Keypoint(x=xmax*0.9, y=ymax-offset),
        Keypoint(x=-1, y=-1),
    ]
    kpsoi = KeypointsOnImage(kps, shape=image.shape)

    # ia.imshow(kpsoi.draw_on_image(image, size=7))
    print(kpsoi.keypoints)

# %% Randomly augment several images
    seq = heavy_augmentation()

    # First element is original image and keypoints
    images_aug = [image]
    kpsois_aug = [kpsoi]

    for i in range(7):
        image_aug, kpsoi_aug = seq(image=image, keypoints=kpsoi)
        images_aug.append(image_aug)
        kpsois_aug.append(kpsoi_aug)

    ia.imshow(
        np.hstack([kpsois_aug[i].draw_on_image(images_aug[i], size=7) for i in range(len(images_aug))])
    )

    # TODO need to check if keypoint is out of image is_out_of_image(image_aug)

    # We _could_ use kpsois_aug[i].clip_out_of_image() to remove all keypoints outside of bounds,
    # but we need keypoint order to manually set heatmaps to 0
# %%

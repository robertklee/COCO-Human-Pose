# %% Import required libraries
# Import utilities
import random

# import the library and helpers
import imageio
import imgaug as ia
import numpy as np  # linear algebra
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmenters.meta import OneOf

from constants import RL_FLIP, ImageAugmentationStrength
from util import validate_enum

# Holy resources: https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B01%20-%20Augment%20Keypoints.ipynb
# Credit to the above notebook for their tutorial on keypoint augmentation

# Not applied transformations but would be interesting to try:
# - iaa.CoarseDropout - Randomly erases a larger chunk of the image - meant to improve robustness for occlusions
# - iaa.SaltAndPepper - Different type of noise

# NOTE: DO NOT APPLY AFFINE SCALE AFTER CROP
# Keypoints that should have disappeared will be part of the image again:
# https://github.com/aleju/imgaug/issues/187

# TODO: It would be nice to begin with a larger image, and then apply rotation / cropping so we get as little 0 padding as possible
# TODO: Verify x,y order in heatmap since the augmentation libary may use a different order

def get_augmenter_pipeline(strength_enum):
    """
    Data augmentation package

    ### Parameters:
    strength_enum : {ImageAugmentationStrength enum}
        Corresponds to the level of data augmentation that should be applied.

    ### Returns:
    Data augmentation pipeline that handles images and corresponding keypoints simultaneously.
    NOTE that both the keypoints and the images must be passed in one call, or different
    transformations will be applied to them, rendering them useless.

    #### Options:
    ImageAugmentationStrength.heavy:
        - Blur with probability 30% with sigma              2.0
        - Up to 3 of the following:
            - Sharpening                                    0.8 to 1.2, blending factor between (0,1)
            - Hue and saturation change of                  -30 to 30 out of 255
            - One of:
                - Brightness change of                      -30 to 30 out of 255
                - Contrast change of                        0.75 to 1.25
            - One of:
                - Pixel dropout up to                       10% of pixels
                - Gaussian noise up to                      5% of 255
        - Scaling                                           (+/- 25 %)
        - Rotation                                          (+/- 30 deg)

    ImageAugmentationStrength.medium:
        - Blur with probability 30% with sigma              1.5
        - Up to 3 of the following:
            - Sharpening                                    0.85 to 1.15, blending factor between (0,0.5)
            - Hue and saturation change of                  -20 to 20 out of 255
            - One of:
                - Brightness change of                      -25 to 25 out of 255
                - Contrast change of                        0.85 to 1.15
            - One of:
                - Pixel dropout up to                       5% of pixels
                - Gaussian noise up to                      3% of 255
        - Scaling                                           (+/- 20 %)
        - Rotation                                          (+/- 25 deg)

    ImageAugmentationStrength.light:
        - Blur with probability 30% with sigma              1
        - Up to 2 of the following:
            - Sharpening                                    None
            - Hue and saturation change of                  -15 to 15 out of 255
            - One of:
                - Brightness change of                      -20 to 20 out of 255
                - Contrast change of                        0.9 to 1.1
            - One of:
                - Pixel dropout up to                       3% of pixels
                - Gaussian noise up to                      1% of 255
        - Scaling                                           (+/- 15 %)
        - Rotation                                          (+/- 15 deg)
    """

    if strength_enum is ImageAugmentationStrength.heavy:
        iaaGaussianBlurSigmaMax                 = 2.0

        iaaMaxNumberImageAppearanceOperations   = 3
        # Image appearance operations
        iaaApplySharpening                      = True
        iaaSharpenAlphaMax                      = 1
        iaaSharpenLightnessMin                  = 0.8
        iaaSharpenLightnessMax                  = 1.2
        iaaHueSaturationMin                     = -30
        iaaHueSaturationMax                     = 30
        iaaBrightnessMin                        = -30
        iaaBrightnessMax                        = 30
        iaaLinearContrastMin                    = 0.75
        iaaLinearContrastMax                    = 1.25
        # Picture quality operations, apply only one
        iaaApplyDropoutAndGaussian              = True
        iaaDropoutPercentPixels                 = 0.1
        iaaAdditiveGaussianNoiseScale           = 0.05

        # Only up to one of the image size/rotation operations
        iaaCropAndPadPercentMagnitude           = 0.25 # add and subtract this from 1.00 (100%, no scaling)
        iaaRotateDegreesMagnitude               = 30 # add and subtract this from 0 deg
    elif strength_enum is ImageAugmentationStrength.medium:
        iaaGaussianBlurSigmaMax                 = 1.5

        iaaMaxNumberImageAppearanceOperations   = 3
        # Image appearance operations
        iaaApplySharpening                      = True
        iaaSharpenAlphaMax                      = 0.5
        iaaSharpenLightnessMin                  = 0.85
        iaaSharpenLightnessMax                  = 1.15
        # Colour and contrast
        iaaHueSaturationMin                     = -20
        iaaHueSaturationMax                     = 20
        iaaBrightnessMin                        = -25
        iaaBrightnessMax                        = 25
        iaaLinearContrastMin                    = 0.85
        iaaLinearContrastMax                    = 1.15
        # Picture quality operations, apply only one
        iaaApplyDropoutAndGaussian              = True
        iaaDropoutPercentPixels                 = 0.05
        iaaAdditiveGaussianNoiseScale           = 0.03

        # Only up to one of the image size/rotation operations
        iaaCropAndPadPercentMagnitude           = 0.20 # add and subtract this from 1.00 (100%, no scaling)
        iaaRotateDegreesMagnitude               = 25 # add and subtract this from 0 deg
    elif strength_enum is ImageAugmentationStrength.light:
        iaaGaussianBlurSigmaMax                 = 1

        iaaMaxNumberImageAppearanceOperations   = 2
        # Image appearance operations
        iaaApplySharpening                      = False
        iaaSharpenAlphaMax                      = 0
        iaaSharpenLightnessMin                  = 1
        iaaSharpenLightnessMax                  = 1
        # Colour and contrast
        iaaHueSaturationMin                     = -15
        iaaHueSaturationMax                     = 15
        iaaBrightnessMin                        = -20
        iaaBrightnessMax                        = 20
        iaaLinearContrastMin                    = 0.9
        iaaLinearContrastMax                    = 1.1
        # Picture quality operations, apply only one
        iaaApplyDropoutAndGaussian              = True
        iaaDropoutPercentPixels                 = 0.03
        iaaAdditiveGaussianNoiseScale           = 0.01

        # Only up to one of the image size/rotation operations
        iaaCropAndPadPercentMagnitude           = 0.15 # add and subtract this from 1.00 (100%, no scaling)
        iaaRotateDegreesMagnitude               = 15 # add and subtract this from 0 deg
    elif strength_enum is ImageAugmentationStrength.none:
        return None
    else:
        validate_enum(ImageAugmentationStrength, strength_enum.name)
        exit(1)

    # Verify that min are lower than max
    assert iaaSharpenLightnessMin   <= iaaSharpenLightnessMax
    assert iaaHueSaturationMin      <= iaaHueSaturationMax
    assert iaaBrightnessMin         <= iaaBrightnessMax
    assert iaaLinearContrastMin     <= iaaLinearContrastMax
    # Verify that the magnitude is given
    assert iaaCropAndPadPercentMagnitude >= 0
    assert iaaRotateDegreesMagnitude     >= 0

    iaaSomeOfImageAppearance = [
            # change their color
            iaa.AddToHueAndSaturation((iaaHueSaturationMin, iaaHueSaturationMax)),
            iaa.OneOf([
                iaa.AddToBrightness((iaaBrightnessMin,iaaBrightnessMax)),
                iaa.LinearContrast((iaaLinearContrastMin, iaaLinearContrastMax)),
            ]),
        ]

    # Conditionally add the following transformations
    if iaaApplySharpening:
        iaaSomeOfImageAppearance.append(
            iaa.Sharpen(alpha=(0, iaaSharpenAlphaMax), lightness=(iaaSharpenLightnessMin, iaaSharpenLightnessMax)),
        )
    if iaaApplyDropoutAndGaussian:
        # Only apply one of the following because they may overlap and do the same thing
        iaaSomeOfImageAppearance.append(iaa.OneOf([
            # randomly remove up to x % of the pixels
            iaa.Dropout((0, iaaDropoutPercentPixels), per_channel=0.5),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, iaaAdditiveGaussianNoiseScale*255), per_channel=0.2),
        ]))

    # define an augmentation pipeline
    aug_pipeline = iaa.Sequential([
        # apply Gaussian blur with a sigma between 0 and x to 30% of the images
        iaa.Sometimes(0.3, iaa.GaussianBlur((0, iaaGaussianBlurSigmaMax))),
        iaa.SomeOf((0, iaaMaxNumberImageAppearanceOperations),
            iaaSomeOfImageAppearance
        ),
        # Only apply one of the following because otherwise there is a risk that keypoints will
        # be pointing to a non-existent part of the image
        iaa.SomeOf((0,1),[
            iaa.CropAndPad(percent=(-1 * iaaCropAndPadPercentMagnitude, iaaCropAndPadPercentMagnitude), keep_size=True, sample_independently=False),
            iaa.Rotate((-1 * iaaRotateDegreesMagnitude, iaaRotateDegreesMagnitude)),
        ])
    ],
    random_order=True # apply the augmentations in random order
    )

    ## Usually, we need this line, but as long as we call the pipeline with both the image and keypoint
    ## passed in together, identical augmentations will be applied to both the image and keypoint
    # aug_pipeline_det = aug_pipeline.to_deterministic()

    return aug_pipeline

def flipRL(image, keypoints, probability=RL_FLIP):
    if random.random() > probability:
        return image, keypoints
    flip = iaa.Fliplr(1.0)
    flipped_img, flipped_kps = flip(image=image, keypoints=keypoints)
    flat_map = lambda f, xs: [y for ys in xs for y in f(ys)] # https://dev.to/turbaszek/flat-map-in-python-3g98
    flipped_fixed_kps = [flipped_kps[0]] + flat_map(lambda pair : (pair[1],pair[0]), zip(flipped_kps[1::2],flipped_kps[2::2]))
    # Leave nose unchanged, flip every pair of eye, ear, etc...
    return flipped_img, flipped_fixed_kps

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
    seq = get_augmenter_pipeline(ImageAugmentationStrength.light)

    # First element is original image and keypoints
    images_aug = [image]
    kpsois_aug = [kpsoi]

    for i in range(7):
        image_aug, kpsoi_aug = seq(image=image, keypoints=kpsoi)

        ## Test flipping function
        # image_aug, kpsoi_aug = flipRL(image=image, keypoints=kpsoi)
        # kpsoi_aug = KeypointsOnImage(kpsoi_aug, shape=image.shape)

        images_aug.append(image_aug)
        kpsois_aug.append(kpsoi_aug)
        # We _could_ use kpsois_aug.clip_out_of_image() to remove all keypoints outside of bounds,
        # but we need keypoint order to manually set heatmaps to 0

    ia.imshow(
        np.hstack([kpsois_aug[i].draw_on_image(images_aug[i], size=7) for i in range(len(images_aug))])
    )
# %%

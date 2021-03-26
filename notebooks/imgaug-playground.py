import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import misc
import imageio
import cv2

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


ia.seed(1)

image = ia.quokka(size=(256, 256))
kps = KeypointsOnImage([
    Keypoint(x=65, y=100),
    Keypoint(x=75, y=200),
    Keypoint(x=100, y=100),
    Keypoint(x=200, y=80)
], shape=image.shape)

seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
    iaa.Affine(
        rotate=10,
        scale=(0.5, 0.7)
    ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
])

# Augment keypoints and images.
image_aug, kps_aug = seq(image=image, keypoints=kps)

# print coordinates before/after augmentation (see below)
# use after.x_int and after.y_int to get rounded integer coordinates
for i in range(len(kps.keypoints)):
    before = kps.keypoints[i]
    after = kps_aug.keypoints[i]
    print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
        i, before.x, before.y, after.x, after.y)
    )

# image with keypoints before/after augmentation (shown below)
image_before = kps.draw_on_image(image, size=7)
image_after = kps_aug.draw_on_image(image_aug, size=7)

def main():
    imgs = np.zeros((1, 100, 100, 3), dtype=np.uint8) + 255
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=0, x2=50, y1=0, y2=50)
    ], shape=imgs.shape[1:])

    aug = iaa.Sequential([
        iaa.Crop(px=10),
        iaa.Pad(px=10, pad_cval=128),
        iaa.Affine(scale=0.5, cval=0)
    ])

    aug_det = aug.to_deterministic()
    imgs_aug = aug_det.augment_images(imgs)
    bbs_aug = aug_det.augment_bounding_boxes([bbs])

    print("bbs:")
    for bbs_aug_i in bbs_aug[0].bounding_boxes:
        print(bbs_aug_i)

    cv2.imshow('orig',imgs)
    cv2.imshow('aug',bbs_aug[0].draw_on_image(imgs_aug[0]))
    cv2.waitKey()

if __name__ == "__main__":
    main()
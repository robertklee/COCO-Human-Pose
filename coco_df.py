from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io

def get_meta(coco):
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        url = img_meta['coco_url']

        yield [img_id, img_file_name, w, h, url, anns]

def convert_to_df(coco, data_set):
    images_data = []
    persons_data = []

    for img_id, img_fname, w, h, url, meta in get_meta(coco):
        images_data.append({
            'image_id': int(img_id),
            'src_set_image_id': int(img_id), # repeat id to reference after join
            'coco_url': url,
            'path': data_set + '/' + img_fname,
            'width': int(w),
            'height': int(h)
        })
        for m in meta:
            persons_data.append({
                'ann_id': m['id'],
                'image_id': m['image_id'],
                'is_crowd': m['iscrowd'],
                'bbox': m['bbox'],
                'bbox_area' : m['bbox'][2] * m['bbox'][3],
                'area': m['area'],
                'num_keypoints': m['num_keypoints'],
                'keypoints': m['keypoints'],
                'segmentation': m['segmentation']
            })

    images_df = pd.DataFrame(images_data)
    images_df.set_index('image_id', inplace=True)

    persons_df = pd.DataFrame(persons_data)
    persons_df.set_index('image_id', inplace=True)

    return images_df, persons_df

def get_df(path_to_train_anns, path_to_val_anns):
    train_coco = COCO(path_to_train_anns) # load annotations for training set
    val_coco = COCO(path_to_val_anns) # load annotations for validation set
    images_df, persons_df = convert_to_df(train_coco, 'train2017')
    train_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
    train_coco_df['source'] = 0
    train_coco_df.head()

    images_df, persons_df = convert_to_df(val_coco, 'val2017')
    val_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
    val_coco_df['source'] = 1
    val_coco_df.head()

    return pd.concat([train_coco_df, val_coco_df], ignore_index=True)
    # ^ Dataframe containing all val and test keypoint annotations
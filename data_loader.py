"""
| *@created on:* 30-06-20,
| *@author:* shubham,
|
| *Description:* loads dataset
"""
import os
import glob
import numpy as np
from PIL import Image


def load_gtea_datasets(ds_path='/Users/subhamsingh/Desktop/gtea_dataset/hand2K_dataset/GTEA/'):
    """
    :param ds_path: path to gtea ds
    :return: loaded orig images and masked images(np format)
    """
    images_path = glob.glob(ds_path + 'Images/*.jpg')
    masks_path = glob.glob(ds_path + 'Masks/*.png')

    images = []
    masks = []

    for img_path, mask_path in zip(images_path, masks_path):
        images.append(np.asarray(Image.open(img_path).resize((128, 128))))
        masks.append(np.asarray(Image.open(mask_path).resize((128, 128))))

    return images, masks


if __name__ == '__main__':
    img, msk = load_gtea_datasets()
    assert np.asarray(img).shape == (663, 128, 128, 3)

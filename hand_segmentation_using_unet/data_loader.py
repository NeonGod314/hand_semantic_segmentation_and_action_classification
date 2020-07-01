"""
| *@created on:* 30-06-20,
| *@author:* shubham,
|
| *Description:* loads dataset
"""
import glob
import numpy as np
from PIL import Image
import tensorflow as tf


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(input_image, input_mask):

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)
    print(input_image.shape, input_mask.shape)

    return input_image, input_mask


def load_gtea_dataset(ds_path='/Users/subhamsingh/Desktop/gtea_dataset/hand2K_dataset/GTEA/'):
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
        mask_colored = Image.open(mask_path).resize((128, 128))
        mask_colored = mask_colored.convert("RGB")
        masks.append(np.asarray(mask_colored))

    images = np.array(images)
    images = images.reshape(images.shape[0], 128, 128, 3)
    masks = np.array(masks)
    masks = masks.reshape(masks.shape[0], 128, 128, 3)

    images, masks = load_image_train(input_image=images, input_mask=masks)

    return images, masks


if __name__ == '__main__':
    img, msk = load_gtea_dataset()
    assert np.asarray(img).shape == (663, 128, 128, 3)

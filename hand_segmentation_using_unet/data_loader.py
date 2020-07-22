"""
| *@created on:* 30-06-20,
| *@author:* shubham,
|
| *Description:* loads dataset
"""
import os
import glob
import numpy as np
import cv2
import tensorflow as tf


def extract_frames_from_video(video_path, save_path):
    video_cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, image = video_cap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(save_path, "frame_{}.jpg".format(count)),
                    image)  # save frame as JPEG file
        count += 1
    return True


def localise_frames(frames_path, save_path):
    frame_path_list = glob.glob(frames_path + '*.jpg')
    count = 0
    for frame_path in frame_path_list:
        raw_frame = cv2.imread(frame_path)
        raw_frame = raw_frame[:, 280:1000]
        cv2.imwrite(save_path + './loacalized_{}.jpg'.format(count), raw_frame)
        count += 1
    return True


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
    return input_image, input_mask


def load_gtea_dataset(ds_path='/Users/subhamsingh/Desktop/gtea_dataset/hand2K_dataset/GTEA/', test_data = False):
    """
    :param ds_path: path to gtea ds
    :return: loaded orig images and masked images(np format)
    """
    images_path = glob.glob(ds_path + 'Images/*.jpg')
    masks_path = ds_path + 'Masks/'

    images = []
    masks = []
    for img_path in images_path:
        img_file_name = os.path.basename(img_path)
        mask_path = (masks_path + img_file_name).split('.')[0] + '.png'
        img = cv2.resize(cv2.imread(img_path), (128, 128))
        msk = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (128, 128))
        ret, msk_binary = cv2.threshold(msk, 20, 255, cv2.THRESH_BINARY)
        images.append(img)
        masks.append(msk_binary)

    images = np.array(images)
    masks = np.array(masks)

    # images, masks = load_image_train(input_image=images, input_mask=masks)

    print("\n#### DATA LOADING COMPLETE\n")
    if test_data:
        test_images = images[:10]
        test_masks = masks[:10]
        return images, masks, test_images, test_masks
    return images, masks


def prediction_pipeline_preprocessor(ds_path):
    images_path = glob.glob(ds_path + '/*.jpg')
    masks_path = ds_path + 'Masks/'

    images = []
    for img_path in images_path:
        img = cv2.resize(cv2.imread(img_path), (128, 128))
        images.append(img)

    images = np.array(images)
    return images


if __name__ == '__main__':
    # extract_frames_from_video(video_path='/Users/subhamsingh/Desktop/cooking data/One-Pot Chicken Fajita Pasta.mp4',
    #                           save_path='/Users/subhamsingh/Desktop/cooking data/raw_frames/')
    localise_frames(frames_path='/Users/subhamsingh/Desktop/cooking data/raw_frames/',
                    save_path='/Users/subhamsingh/Desktop/cooking data/localized_frames/')
    # img, msk = load_gtea_dataset()
    # print(msk[0].shape)
    # print(msk[0])
    # cv2.imwrite('tmp.jpg', np.asarray(msk[0]))
    # cv2.imwrite('tmp2.jpg', np.asarray(img[0]))
    #
    #
    # assert np.asarray(img).shape == (663, 128, 128, 3)
    # print(np.asarray(msk).shape)
    # assert np.asarray(msk).shape == (663, 128, 128)
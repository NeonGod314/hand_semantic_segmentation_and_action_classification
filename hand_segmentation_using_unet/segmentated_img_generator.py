import numpy as np
import cv2
import glob

project_dir = '/Users/subhamsingh/Desktop/hand_semantic_segmentation_and_action_classification/'


def generator(mask, img, save_path):
    mask_indexes = mask < 10
    comb_img = img.copy() * 0
    comb_img[mask_indexes] = img[mask_indexes]
    cv2.imwrite(save_path, comb_img)


if __name__ == '__main__':
    all_orig_images = glob.glob(project_dir + 'version_1_predictions/predict_img_*.jpg')

    for img_path in all_orig_images:
        img = cv2.imread(img_path)
        msk_path = img_path.replace('img', 'msk')
        msk = cv2.imread(msk_path)

        img_num = img_path.split('_img_')[1].split('.0')[0]

        generator(mask=msk, img=img, save_path=project_dir + 'version_1_predictions/seg_' + str(img_num) + '.jpg')

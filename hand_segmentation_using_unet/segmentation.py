# -*- coding: utf-8 -*-
"""
| *@created on:* 1-07-20,
| *@author:* shubham,
|
| *Description:* Basic Unet
"""

import cv2
import numpy as np
from hand_segmentation_using_unet.network import UNET
import tensorflow as tf
from hand_segmentation_using_unet.data_loader import load_gtea_dataset, prediction_pipeline_preprocessor
import matplotlib.pyplot as plt

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        print("i: ", 1)
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset):
    if dataset:
        for image, mask in dataset:
            pred_mask = model.predict(np.array([image]))
            pred_img = pred_mask[0]
            pred_img = pred_img.reshape([128, 128]).astype('uint8')
            plt.imshow(pred_img)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions(dataset=zip(images[1:2], masks[1:2]))
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


if __name__ == '__main__':

    EPOCHS = 50
    BATCH_SIZE = 128
    LR = 0.6
    OUTPUT_CHANNELS = None
    images, masks, test_images, test_masks = load_gtea_dataset(test_data=True)

    images = images / 255.
    test_images = test_images / 255.

    print("train_image shape: ", images.shape)
    print("train_mask shape: ", masks.shape)

    unet = UNET(output_channels=OUTPUT_CHANNELS)
    lr_scheduler = unet.callback

    model = unet.network()
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='model_arch.jpg')
    weights = model.get_weights()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss=tf.keras.losses.MeanSquaredError())

    print("\n###### Model Information")
    for layer in model.layers:
        print(layer.name, layer.output_shape)
        print("____________________________")

    model_history = model.fit(x=images, y=masks, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_scheduler])

    # unet.save_trained_model(
    #     weights_path='/hand_segmentation_using_unet/weights/temp_30.pickle')

    ## Uncomment following part for loading trained model
    """
    new_model = unet.load_model(
        weights_path='/Users/subhamsingh/Desktop/hand_semantic_segmentation_and_action_classification/hand_segmentation'
                     '_using_unet/weights/50_epoch_train.pickle')
    print(new_model)
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.006),
                  loss=tf.keras.losses.MeanSquaredError())

    new_model_history = new_model.fit(x=images, y=masks, epochs=EPOCHS, batch_size=BATCH_SIZE)
    """
    predicts = model.predict(test_images[:10])
    count = 0
    import json

    for pred in predicts:
        pred = pred.reshape([128, 128]).astype('uint8')
        cv2.imshow('hello', pred)
        cv2.waitKey(5000)
        cv2.imwrite('./pred_' + str(count) + '.jpg', pred)
        count += 1
    print(json.dumps(predicts[0]))

    predict_dataset = zip(test_images[:10], test_masks[:10])
    show_predictions(dataset=zip(images[0:2], masks[0:2]))

    show_predictions(dataset=zip(images[0:2], masks[0:2]))
    loss = model_history.history['loss']

    print("loss: ", model_history.history)
    epochs = range(EPOCHS)
    show_predictions(dataset=zip(images[1:2], masks[1:2]))

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

# -*- coding: utf-8 -*-
"""
| *@created on:* 1-07-20,
| *@author:* shubham,
|
| *Description:* Basic Unet
"""

import cv2
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow_examples.models.pix2pix import pix2pix
from hand_segmentation_using_unet.data_loader import load_gtea_dataset
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


OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

"""The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples."""

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        1, 3, strides=2,
        padding='same')  # 64x64 -> 128x128

    last_2 = tf.keras.layers.Reshape(target_shape=[128, 128])
    x = last(x)
    x = last_2(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.006),
              loss=tf.keras.losses.MeanSquaredError())


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset, num=1, epoch=0):
    if dataset:
        for image, mask in dataset:
            print("input image shape: ", image.shape, "mask shape: ", mask.shape)
            pred_mask = model.predict(np.array([image]))
            print("predicted mask shape: ", pred_mask[0].shape)
            pred_img = pred_mask[0]
            pred_img = pred_img.reshape([128, 128]).astype('uint8')
            print(pred_img.shape)
            # cv2.imshow("predicted image", pred_img)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # clear_output(wait=True)
        show_predictions(dataset=zip(images[1:2], masks[1:2]), epoch=epoch)
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


if __name__ == '__main__':

    EPOCHS = 80
    BATCH_SIZE = 64
    images, masks, test_images, test_masks = load_gtea_dataset(test_data=True)

    print("train_image shape: ", images.shape)
    print("train_mask shape: ", masks.shape)

    print("\n#### Model Information\n")
    for layer in model.layers:
        print(layer.name, layer.output_shape)

    steps_per_epoch = len(images) // EPOCHS
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    print(tf.trainable_variables())
    exit()
    model_history = model.fit(x=images, y=masks, epochs=EPOCHS, batch_size=BATCH_SIZE)

    predict_dataset = zip(test_images, test_masks)

    i = 0
    for img, msk in predict_dataset:
        cv2.imwrite('predict_img_' + str(i) + '.jpg', np.array(img))
        cv2.imwrite('predict_msk_' + str(i) + '.jpg', np.array(msk))
        i = i + 1

    show_predictions(dataset=zip(images[0:2], masks[0:2]))

    exit()
    model.save(
        filepath='/Users/subhamsingh/Desktop/hand_semantic_segmentation_and_action_classification/saved_models/basic_unet',
        overwrite=True)

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

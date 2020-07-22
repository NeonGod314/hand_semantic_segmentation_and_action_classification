import numpy as np
import tensorflow as tf
import pickle

class InstanceNormalization(tf.keras.layers.Layer):
    """
    Custom keras layer implementation for Instance Normalization
    normalizes across spatial locations only
    """

    def __init__(self, name, epsilon=1e-5):
        super(InstanceNormalization, self).__init__(name=name)
        self.epsilon = epsilon
        self.scale = None
        self.offset = None

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class UNET:
    def __init__(self, output_channels):
        self.unet_model = None
        self.output_channels = output_channels
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

        # down stack layers

        # layer creation
        self.layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]

        # up sample layers
        self.up_sample_1 = self.up_sample(filters=512, size=3, name='up_1')  # 4x4 -> 8x8
        self.up_sample_2 = self.up_sample(filters=256, size=3, name='up_2')  # 8x8 -> 16x16
        self.up_sample_3 = self.up_sample(filters=128, size=3, name='up_3')  # 16x16 -> 32x32
        self.up_sample_4 = self.up_sample(filters=64, size=3, name='up_4')  # 32x32 -> 64x64

        self.last_layer = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same',
                                                          name='last_layer')  # 64x64 -> 128x128

        if self.output_channels is None or self.output_channels == 0:
            self.last_2 = tf.keras.layers.Reshape(target_shape=[128, 128])
        else:
            self.last_2 = tf.keras.layers.Reshape(target_shape=[128, 128, self.output_channels])

        self.callback = tf.keras.callbacks.LearningRateScheduler(UNET.scheduler)

    def network(self):
        """
        Custom UNET model
        Down stack is Mobilenet which remains non trainable throughout
        :param output_channels: Number of output channles for model
        :return: Model
        """

        layers = [self.base_model.get_layer(name).output for name in self.layer_names]
        down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=layers)
        down_stack.trainable = False

        up_stack = [
            self.up_sample_1,
            self.up_sample_2,
            self.up_sample_3,
            self.up_sample_4
        ]

        inputs = tf.keras.layers.Input(shape=[128, 128, 3])

        # structure creation
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

        x = self.last_layer(x)
        x = self.last_2(x)
        self.unet_model = tf.keras.Model(inputs=inputs, outputs=x)
        return self.unet_model

    def save_trained_model(self, weights_path):
        """
        Saves the weights of the model
        :param weights_path: path to which save. should include the name for pickle file as well
        :return: --
        """
        model_weights = {}
        trainable_up_sample_layers = ['up_1', 'up_2', 'up_3', 'up_4']
        upsample_intermideates = ['_c2t', '_bn']
        last_layer_name = 'last_layer'

        # get upsampling layer weights
        layers = self.unet_model.layers
        for tens in layers:
            print(tens.__class__)
        for up_sample_layer in trainable_up_sample_layers:
            for intermediate in upsample_intermideates:
                layer_name = up_sample_layer + intermediate

                model_weights[layer_name] = self.unet_model.get_layer(up_sample_layer).get_layer(
                    layer_name).get_weights()

        # get last layer weight
        model_weights[last_layer_name] = self.unet_model.get_layer(last_layer_name).get_weights()

        with open(weights_path, 'wb') as handle:
            pickle.dump(model_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return True

    def load_model(self, weights_path):
        with open(weights_path, 'rb') as handle:
            print(weights_path)
            weights_dictionary = pickle.load(handle)

        layers = [self.base_model.get_layer(name).output for name in self.layer_names]
        down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=layers)
        down_stack.trainable = False

        up_stack = [
            self.up_sample(filters=512, size=3, name='up_1', weights=weights_dictionary),  # 4x4 -> 8x8
            self.up_sample(filters=256, size=3, name='up_2', weights=weights_dictionary),  # 8x8 -> 16x16
            self.up_sample(filters=128, size=3, name='up_3', weights=weights_dictionary),  # 16x16 -> 32x32
            self.up_sample(filters=64, size=3, name='up_4', weights=weights_dictionary)  # 32x32 -> 64x64

        ]

        inputs = tf.keras.layers.Input(shape=[128, 128, 3])

        # structure creation
        x = inputs

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            print(up)
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        self.last_layer = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', name='last_layer',
                                                          weights=weights_dictionary['last_layer'])  # 64x64 -> 128x128

        print(self.last_layer)

        x = self.last_layer(x)
        x = self.last_2(x)
        self.unet_model = tf.keras.Model(inputs=inputs, outputs=x)
        return self.unet_model

    def up_sample(self, name, filters, size, norm_type='batchnorm', apply_dropout=False, weights=None):
        """
        Upsamples an input.
        Conv2DTranspose => Batchnorm => Dropout => Relu
        :param name: name of component
        :param filters: Number of filters
        :param size: Filter size / [filter size]*2
        :param norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        :param apply_dropout: If True, adds the dropout layer
        :return: Upsample Sequential Model
        """
        conv2Dtranspose_name = name + '_c2t'
        batch_normalization_name = name + '_bn'
        instance_normalization_name = name + '_in'

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential(name=name)
        if weights:
            conv2d_transpose = tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                               padding='same', name=conv2Dtranspose_name,
                                                               weights=weights[conv2Dtranspose_name], use_bias=False)
        else:
            conv2d_transpose = tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                               padding='same',
                                                               kernel_initializer=initializer,
                                                               use_bias=False, name=conv2Dtranspose_name)

        result.add(conv2d_transpose)

        if norm_type.lower() == 'batchnorm':
            if weights:
                batch_normalization = tf.keras.layers.BatchNormalization(name=batch_normalization_name,
                                                                         weights=weights[batch_normalization_name])
            else:
                batch_normalization = tf.keras.layers.BatchNormalization(name=batch_normalization_name)

            result.add(batch_normalization)
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization(name=instance_normalization_name))

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    @classmethod
    def scheduler(cls, epoch):
        if epoch < 10:
            return 0.09
        elif 10 < epoch < 15:
            return 0.06
        elif 15 < epoch < 20:
            return 0.03
        elif 20 < epoch < 30:
            return 0.006
        elif 30 < epoch < 55:
            return 0.003
        else:
            return 0.001


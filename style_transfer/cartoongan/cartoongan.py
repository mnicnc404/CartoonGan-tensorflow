import os
import numpy as np
import tensorflow as tf
from keras_contrib.layers import InstanceNormalization
from tensorflow.keras.layers import Layer, InputSpec

PRETRAINED_WEIGHT_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pretrained_weights")


# ref: https://stackoverflow.com/a/53349976/2447655
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


def conv_layer(style, name, filters, kernel_size, strides=(1, 1), bias=True):
    init_weight = np.load(f"{PRETRAINED_WEIGHT_DIR}/{style}/{name}.weight.npy")
    init_weight = np.transpose(init_weight, [2, 3, 1, 0])
    init_bias = np.load(f"{PRETRAINED_WEIGHT_DIR}/{style}/{name}.bias.npy")

    if bias:
        bias_initializer = tf.keras.initializers.constant(init_bias)
    else:
        bias_initializer = "zeros"

    layer = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=tf.keras.initializers.constant(init_weight),
        bias_initializer=bias_initializer
    )
    return layer


def instance_norm_layer(style, name, epsilon=1e-9):
    init_beta = np.load(f"{PRETRAINED_WEIGHT_DIR}/{style}/{name}.shift.npy")
    init_gamma = np.load(f"{PRETRAINED_WEIGHT_DIR}/{style}/{name}.scale.npy")

    layer = InstanceNormalization(
        axis=-1,
        epsilon=epsilon,
        beta_initializer=tf.keras.initializers.constant(init_beta),
        gamma_initializer=tf.keras.initializers.constant(init_gamma)
    )
    return layer


def deconv_layers(style, name, filters, kernel_size, strides=(1, 1)):
    init_weight = np.load(f"{PRETRAINED_WEIGHT_DIR}/{style}/{name}.weight.npy")
    init_weight = np.transpose(init_weight, [2, 3, 1, 0])
    init_bias = np.load(f"{PRETRAINED_WEIGHT_DIR}/{style}/{name}.bias.npy")

    layers = list()
    layers.append(tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=tf.keras.initializers.constant(init_weight),
        bias_initializer=tf.keras.initializers.constant(init_bias)
    ))

    layers.append(tf.keras.layers.Cropping2D(cropping=((1, 0), (1, 0))))
    return layers


def load_model(style):
    inputs = tf.keras.Input(shape=(None, None, 3))

    # y = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(inputs)
    y = ReflectionPadding2D(padding=(3, 3))(inputs)
    y = conv_layer(style, "conv01_1", filters=64, kernel_size=7)(y)
    y = instance_norm_layer(style, "in01_1")(y)
    y = tf.keras.layers.Activation("relu")(y)

    # y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
    y = ReflectionPadding2D(padding=(1, 1))(y)
    y = conv_layer(style, "conv02_1", filters=128, kernel_size=3, strides=(2, 2))(y)
    # y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
    y = ReflectionPadding2D(padding=(1, 1))(y)
    y = conv_layer(style, "conv02_2", filters=128, kernel_size=3, strides=(1, 1))(y)
    y = instance_norm_layer(style, "in02_1")(y)
    y = tf.keras.layers.Activation("relu")(y)

    # y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
    y = ReflectionPadding2D(padding=(1, 1))(y)
    y = conv_layer(style, "conv03_1", filters=256, kernel_size=3, strides=(2, 2))(y)
    # y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
    y = ReflectionPadding2D(padding=(1, 1))(y)
    y = conv_layer(style, "conv03_2", filters=256, kernel_size=3, strides=(1, 1))(y)
    y = instance_norm_layer(style, "in03_1")(y)

    t_prev = tf.keras.layers.Activation("relu")(y)

    for i in range(4, 12):
        # y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(t_prev)
        y = ReflectionPadding2D(padding=(1, 1))(t_prev)
        y = conv_layer(style, "conv%02d_1" % i, filters=256, kernel_size=3)(y)
        y = instance_norm_layer(style, "in%02d_1" % i)(y)
        y = tf.keras.layers.Activation("relu")(y)

        t = ReflectionPadding2D(padding=(1, 1))(y)
        t = conv_layer(style, "conv%02d_2" % i, filters=256, kernel_size=3)(t)
        t = instance_norm_layer(style, "in%02d_2" % i)(t)

        t_prev = tf.keras.layers.Add()([t, t_prev])

        if i == 11:
            y = t_prev

    layers = deconv_layers(style, "deconv01_1", filters=128, kernel_size=3, strides=(2, 2))
    for layer in layers:
        y = layer(y)
    # y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
    y = ReflectionPadding2D(padding=(1, 1))(y)
    y = conv_layer(style, "deconv01_2", filters=128, kernel_size=3)(y)
    y = instance_norm_layer(style, "in12_1")(y)
    y = tf.keras.layers.Activation("relu")(y)

    layers = deconv_layers(style, "deconv02_1", filters=64, kernel_size=3, strides=(2, 2))
    for layer in layers:
        y = layer(y)
    # y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
    y = ReflectionPadding2D(padding=(1, 1))(y)
    y = conv_layer(style, "deconv02_2", filters=64, kernel_size=3)(y)
    y = instance_norm_layer(style, "in13_1")(y)
    y = tf.keras.layers.Activation("relu")(y)

    # y = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(y)
    y = ReflectionPadding2D(padding=(3, 3))(y)
    y = conv_layer(style, "deconv03_1", filters=3, kernel_size=7)(y)
    y = tf.keras.layers.Activation("tanh")(y)

    model = tf.keras.Model(inputs=inputs, outputs=y)

    return model


if __name__ == '__main__':
    g = load_model(style="shinkai")
    np.random.seed(9527)
    nx = np.random.rand(1, 225, 225, 3).astype(np.float32)
    out = g(nx)
    tf_out = np.load("tf_out.npy")

    diff = np.sqrt(np.mean((out - tf_out) ** 2))
    assert diff < 1e-6

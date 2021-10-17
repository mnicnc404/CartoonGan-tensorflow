import os
import numpy as np
import tensorflow as tf
# from keras_contrib.layers import InstanceNormalization
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.layers import Layer, InputSpec

PRETRAINED_WEIGHT_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pretrained_weights")


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=3,
                 epsilon=1e-9,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = tf.keras.layers.InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = tf.keras.backend.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = tf.keras.backend.mean(inputs, reduction_axes, keepdims=True)
        stddev = tf.keras.backend.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = tf.keras.backend.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = tf.keras.backend.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
        beta_initializer=tf.keras.initializers.Constant(init_beta),
        gamma_initializer=tf.keras.initializers.Constant(init_gamma)
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

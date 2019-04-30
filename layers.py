import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, InputSpec, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add
from tensorflow.keras.layers import ReLU, LeakyReLU, ZeroPadding2D
from keras_contrib.layers import InstanceNormalization


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        padding = tuple(padding)
        self.padding = ((0, 0), padding, padding, (0, 0))
        self.input_spec = [InputSpec(ndim=4)]

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x):
        return tf.pad(x, self.padding, "REFLECT")


def get_padding(pad_type, padding):
    if pad_type == "reflect":
        return ReflectionPadding2D(padding)
    elif pad_type == "constant":
        return ZeroPadding2D(padding)
    else:
        raise ValueError(f"Unrecognized pad_type {pad_type}")


def get_norm(norm_type):
    if norm_type == "instance":
        return InstanceNormalization()
    elif norm_type == 'batch':
        return BatchNormalization()
    else:
        raise ValueError(f"Unrecognized norm_type {norm_type}")


class LightConv(Model):
    def __init__(self,
                 mid_filters,
                 filters,
                 kernel_size,
                 stride,
                 norm_type="instance",
                 pad_type="constant",
                 **kwargs):
        super(LightConv, self).__init__(name="FlatConv")
        padding = (kernel_size - 1) // 2
        padding = (padding, padding)
        self.model = tf.keras.models.Sequential()
        self.model.add(Conv2D(mid_filters, 1, use_bias=False))
        self.model.add(get_norm(norm_type))
        self.model.add(ReLU())
        self.model.add(get_padding(pad_type, padding))
        self.model.add(DepthwiseConv2D(3, stride, use_bias=False))
        self.model.add(get_norm(norm_type))
        self.model.add(ReLU())
        self.model.add(Conv2D(filters, 1))

    def build(self, input_shape):
        super(LightConv, self).build(input_shape)

    def call(self, x, training=False):
        return self.model(x, training=training)


class FlatConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 norm_type="instance",
                 pad_type="constant",
                 **kwargs):
        super(FlatConv, self).__init__(name="FlatConv")
        padding = (kernel_size - 1) // 2
        padding = (padding, padding)
        self.model = tf.keras.models.Sequential()
        self.model.add(get_padding(pad_type, padding))
        self.model.add(Conv2D(filters, kernel_size))
        self.model.add(get_norm(norm_type))
        self.model.add(ReLU())

    def build(self, input_shape):
        super(FlatConv, self).build(input_shape)

    def call(self, x, training=False):
        return self.model(x, training=training)


class ConvBlock(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride=1,
                 norm_type="instance",
                 pad_type="constant",
                 **kwargs):
        super(ConvBlock, self).__init__(name="ConvBlock")
        padding = (kernel_size - 1) // 2
        padding = (padding, padding)

        self.model = tf.keras.models.Sequential()
        self.model.add(get_padding(pad_type, padding))
        self.model.add(Conv2D(filters, kernel_size, stride))
        self.model.add(get_padding(pad_type, padding))
        self.model.add(Conv2D(filters, kernel_size))
        self.model.add(get_norm(norm_type))
        self.model.add(ReLU())

    def build(self, input_shape):
        super(ConvBlock, self).build(input_shape)

    def call(self, x, training=False):
        return self.model(x, training=training)


class ResBlock(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 norm_type="instance",
                 pad_type="constant",
                 light=False,
                 **kwargs):
        super(ResBlock, self).__init__(name="ResBlock")
        padding = (kernel_size - 1) // 2
        padding = (padding, padding)
        self.model = tf.keras.models.Sequential()
        if light:
            self.model.add(LightConv(
                filters, filters, kernel_size, 1, norm_type, pad_type))
        else:
            self.model.add(get_padding(pad_type, padding))
            self.model.add(Conv2D(filters, kernel_size))
            self.model.add(get_norm(norm_type))
            self.model.add(ReLU())
            self.model.add(get_padding(pad_type, padding))
            self.model.add(Conv2D(filters, kernel_size))
            self.model.add(get_norm(norm_type))
        self.add = Add()

    def build(self, input_shape):
        super(ResBlock, self).build(input_shape)

    def call(self, x, training=False):
        return self.add([self.model(x, training=training), x])


class UpSampleConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 norm_type="instance",
                 pad_type="constant",
                 light=False,
                 **kwargs):
        super(UpSampleConv, self).__init__(name="UpSampleConv")
        if light:
            self.model = LightConv(
                filters, filters, kernel_size, 1, norm_type, pad_type)
        else:
            self.model = ConvBlock(
                filters, kernel_size, 1, norm_type, pad_type)

    def build(self, input_shape):
        super(UpSampleConv, self).build(input_shape)

    def call(self, x, training=False):
        cur_h = x.shape[1] // 2 * 4  # NOTE: avoid shape of odd nums
        cur_w = x.shape[2] // 2 * 4
        x = tf.image.resize(x, (cur_h, cur_w))
        return self.model(x, training=training)


class StridedConv(Model):
    def __init__(self,
                 filters=64,
                 lrelu_alpha=0.2,
                 pad_type="constant",
                 norm_type="batch",
                 **kwargs):
        super(StridedConv, self).__init__(name="StridedConv")

        self.model = tf.keras.models.Sequential()
        self.model.add(get_padding(pad_type, (1, 1)))
        self.model.add(Conv2D(filters, 3, strides=(2, 2)))
        self.model.add(LeakyReLU(lrelu_alpha))
        self.model.add(get_padding(pad_type, (1, 1)))
        self.model.add(Conv2D(filters * 2, 3))
        self.model.add(get_norm(norm_type))
        self.model.add(LeakyReLU(lrelu_alpha))

    def build(self, input_shape):
        super(StridedConv, self).build(input_shape)

    def call(self, x, training=False):
        return self.model(x, training=training)

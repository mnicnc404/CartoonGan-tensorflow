import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, InputSpec
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
                 in_filters,
                 filters,
                 kernel_size,
                 **kwargs):
        pass


class FlatConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 norm_type="instance",
                 pad_type="constant",
                 **kwargs):
        super(FlatConv, self).__init__(name="FlatConv")
        self.model = tf.keras.models.Sequential()
        padding = (kernel_size - 1) // 2
        padding = (padding, padding)
        self.model.add(get_padding(pad_type, padding))
        self.model.add(Conv2D(filters, kernel_size))
        self.model.add(get_norm(norm_type))
        self.model.add(ReLU())

    def build(self, input_shape):
        super(FlatConv, self).build(input_shape)

    def call(self, x, training=False):
        return self.model(x, training=training)


class DownSampleConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 norm_type="instance",
                 pad_type="constant",
                 **kwargs):
        super(DownSampleConv, self).__init__(name="DownSampleConv")
        padding = (kernel_size - 1) // 2
        padding = (padding, padding)

        self.model = tf.keras.models.Sequential()
        self.model.add(get_padding(pad_type, padding))
        self.model.add(Conv2D(filters, kernel_size, 2))
        self.model.add(get_padding(pad_type, padding))
        self.model.add(Conv2D(filters, kernel_size))
        self.model.add(get_norm(norm_type))
        self.model.add(ReLU())

    def build(self, input_shape):
        super(DownSampleConv, self).build(input_shape)

    def call(self, x, training=False):
        return self.model(x, training=training)


class ResBlock(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 norm_type="instance",
                 pad_type="constant",
                 **kwargs):
        super(ResBlock, self).__init__(name="ResBlock")
        padding = (kernel_size - 1) // 2
        padding = (padding, padding)
        self.model = tf.keras.models.Sequential()
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
                 **kwargs):
        super(UpSampleConv, self).__init__(name="UpSampleConv")
        self.model = tf.keras.models.Sequential()
        self.model.add(Conv2D(filters, kernel_size, padding="same"))
        self.model.add(Conv2D(filters, kernel_size, padding="same"))
        self.model.add(get_norm(norm_type))
        self.model.add(ReLU())

    def build(self, input_shape):
        super(UpSampleConv, self).build(input_shape)

    def call(self, x, training=False):
        cur_h = x.shape[1] // 2 * 4
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

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


class FlatConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 padding=(1, 1),
                 norm_type="instance",
                 pad_type="reflect"):
        super(FlatConv, self).__init__(name="FlatConv")
        self.model = tf.keras.models.Sequential()

        if pad_type == "reflect":
            self.model.add(ReflectionPadding2D(padding))
        elif pad_type == "constant":
            self.model.add(ZeroPadding2D(padding))
        else:
            raise ValueError(f"Unrecognized pad_type {pad_type}")

        self.model.add(Conv2D(filters, kernel_size))

        if norm_type == "instance":
            self.model.add(InstanceNormalization())
        elif norm_type == "batch":
            self.model.add(BatchNormalization())
        else:
            raise ValueError(f"Unrecognized norm_type {norm_type}")

        self.model.add(ReLU())

    def build(self, input_shape):
        super(FlatConv, self).build(input_shape)

    def call(self, x, training=False):
        return self.model(x)


class DownSampleConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 padding=(1, 1),
                 norm_type="instance",
                 pad_type="reflect"):
        super(DownSampleConv, self).__init__(name="DownSampleConv")
        if pad_type == "reflect":
            padder = ReflectionPadding2D
        elif pad_type == "constant":
            padder = ZeroPadding2D
        else:
            raise ValueError(f"Unrecognized pad_type {pad_type}")

        self.model = tf.keras.models.Sequential()
        self.model.add(padder(padding))
        self.model.add(Conv2D(filters, kernel_size, 2))
        self.model.add(padder(padding))
        self.model.add(Conv2D(filters, kernel_size))

        if norm_type == "instance":
            self.model.add(InstanceNormalization())
        elif norm_type == "batch":
            self.model.add(BatchNormalization())
        else:
            raise ValueError(f"Unrecognized norm_type {norm_type}")

        self.model.add(ReLU())

    def build(self, input_shape):
        super(DownSampleConv, self).build(input_shape)

    def call(self, x, training=False):
        return self.model(x)


class ResBlock(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 padding=(1, 1),
                 norm_type="instance",
                 pad_type="reflect"):
        super(ResBlock, self).__init__(name="ResBlock")
        if pad_type == "reflect":
            padder = ReflectionPadding2D
        elif pad_type == "constant":
            padder = ZeroPadding2D
        else:
            raise ValueError(f"Unrecognized pad_type {pad_type}")

        if norm_type == "instance":
            normer = InstanceNormalization
        elif norm_type == "batch":
            normer = BatchNormalization
        else:
            raise ValueError(f"Unrecognized norm_type {norm_type}")

        self.model = tf.keras.models.Sequential()
        self.model.add(padder(padding))
        self.model.add(Conv2D(filters, kernel_size))
        self.model.add(normer())
        self.model.add(ReLU())
        self.model.add(padder(padding))
        self.model.add(Conv2D(filters, kernel_size))
        self.model.add(normer())
        self.add = Add()

    def build(self, input_shape):
        super(ResBlock, self).build(input_shape)

    def call(self, x, training=False):
        return self.add([self.model(x), x])


class UpSampleConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 norm_type="instance"):
        super(UpSampleConv, self).__init__(name="UpSampleConv")
        self.model = tf.keras.models.Sequential()
        self.model.add(Conv2D(filters, kernel_size, padding="same"))
        self.model.add(Conv2D(filters, kernel_size, padding="same"))

        if norm_type == "instance":
            self.model.add(InstanceNormalization())
        elif norm_type == "batch":
            self.model.add(BatchNormalization())
        else:
            raise ValueError(f"Unrecognized norm_type {norm_type}")

        self.model.add(ReLU())

    def build(self, input_shape):
        super(UpSampleConv, self).build(input_shape)

    def call(self, x, training=False):
        cur_h = x.shape[1] // 2 * 4
        cur_w = x.shape[2] // 2 * 4
        x = tf.image.resize(x, (cur_h, cur_w))
        return self.model(x)


class StridedConv(Model):
    def __init__(self,
                 filters=64,
                 lrelu_alpha=0.2,
                 pad_type="reflect",
                 norm_type="batch"):
        super(StridedConv, self).__init__(name="StridedConv")
        if pad_type == "reflect":
            padder = ReflectionPadding2D
        elif pad_type == "constant":
            padder = ZeroPadding2D
        else:
            raise ValueError(f"Unrecognized pad_type {pad_type}")

        self.model = tf.keras.models.Sequential()
        self.model.add(padder())
        self.model.add(Conv2D(filters, 3, strides=(2, 2)))
        self.model.add(LeakyReLU(lrelu_alpha))
        self.model.add(padder())
        self.model.add(Conv2D(filters * 2, 3))

        if norm_type == "instance":
            self.model.add(InstanceNormalization())
        elif norm_type == "batch":
            self.model.add(BatchNormalization())
        else:
            raise ValueError(f"Unrecognized norm_type {norm_type}")

        self.model.add(LeakyReLU(lrelu_alpha))

    def build(self, input_shape):
        super(StridedConv, self).build(input_shape)

    def call(self, x, training=False):
        return self.model(x)

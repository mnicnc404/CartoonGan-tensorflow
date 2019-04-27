import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
    BatchNormalization, ReLU, LeakyReLU, ZeroPadding2D, Add
from keras_contrib.layers import InstanceNormalization


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], "REFLECT")


class FlatConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 padding=(1, 1),
                 norm_type="instance",
                 pad_type="reflect"):
        super(FlatConv, self).__init__(name="FlatConv")
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.norm_type = norm_type
        self.pad_type = pad_type

        if self.pad_type == "reflect":
            self.pad = ReflectionPadding2D(self.padding)
        elif self.pad_type == "constant":
            self.pad = ZeroPadding2D(self.padding)

        self.conv = Conv2D(self.filters, self.kernel_size)

        if self.norm_type == "instance":
            self.norm = InstanceNormalization()
        elif self.norm_type == "batch":
            self.norm = BatchNormalization()

        self.act = ReLU()

    def build(self, input_shape):
        super(FlatConv, self).build(input_shape)

    def call(self, x, training=False):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = self.act(x)
        return x


class DownSampleConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 padding=(1, 1),
                 norm_type="instance",
                 pad_type="reflect"):
        super(DownSampleConv, self).__init__(name="DownSampleConv")
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.norm_type = norm_type
        self.pad_type = pad_type

        if self.pad_type == "reflect":
            self.pad1 = ReflectionPadding2D(self.padding)
        elif self.pad_type == "constant":
            self.pad1 = ZeroPadding2D(self.padding)
        self.conv1 = Conv2D(self.filters, self.kernel_size, 2)

        if self.pad_type == "reflect":
            self.pad2 = ReflectionPadding2D(self.padding)
        elif self.pad_type == "constant":
            self.pad2 = ZeroPadding2D(self.padding)
        self.conv2 = Conv2D(self.filters, self.kernel_size)

        if self.norm_type == "instance":
            self.norm = InstanceNormalization()
        elif self.norm_type == "batch":
            self.norm = BatchNormalization()

        self.act = ReLU()

    def build(self, input_shape):
        super(DownSampleConv, self).build(input_shape)

    def call(self, x, training=False):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm(x, training=training)
        x = self.act(x)
        return x


class ResBlock(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 padding=(1, 1),
                 norm_type="instance",
                 pad_type="reflect"):
        super(ResBlock, self).__init__(name="ResBlock")
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.norm_type = norm_type
        self.pad_type = pad_type

        if self.pad_type == "reflect":
            self.pad1 = ReflectionPadding2D(self.padding)
        elif self.pad_type == "constant":
            self.pad1 = ZeroPadding2D(self.padding)

        self.conv1 = Conv2D(self.filters, self.kernel_size)

        if self.norm_type == "instance":
            self.norm1 = InstanceNormalization()
        elif self.norm_type == "batch":
            self.norm1 = BatchNormalization()

        self.act = ReLU()

        if self.pad_type == "reflect":
            self.pad2 = ReflectionPadding2D(self.padding)
        elif self.pad_type == "constant":
            self.pad2 = ZeroPadding2D(self.padding)

        self.conv2 = Conv2D(self.filters, self.kernel_size)

        if self.norm_type == "instance":
            self.norm2 = InstanceNormalization()
        elif self.norm_type == "batch":
            self.norm2 = BatchNormalization()

        self.add = Add()

    def build(self, input_shape):
        super(ResBlock, self).build(input_shape)

    def call(self, x, training=False):
        x_prev = x
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.act(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.add([x, x_prev])
        return x


class UpSampleConv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 norm_type="instance"):
        super(UpSampleConv, self).__init__(name="UpSampleConv")
        self.filters = filters
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        self.deconv1 = Conv2D(self.filters,
                              self.kernel_size,
                              padding="same")
        self.deconv2 = Conv2D(self.filters,
                              self.kernel_size,
                              padding="same")

        if self.norm_type == "instance":
            self.norm = InstanceNormalization()
        elif self.norm_type == "batch":
            self.norm = BatchNormalization()

        self.act = ReLU()

    def build(self, input_shape):
        super(UpSampleConv, self).build(input_shape)

    def call(self, x, training=False):
        cur_h = x.shape[1] // 2 * 4
        cur_w = x.shape[2] // 2 * 4
        boxes = [[0, 0, 1, 1]] * x.shape[0]
        box_indices = list(range(x.shape[0]))
        # ref: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/image/crop_and_resize
        x = tf.image.crop_and_resize(x, boxes, box_indices, (cur_h, cur_w))

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.norm(x, training=training)
        x = self.act(x)
        return x


class StridedConv(Model):
    def __init__(self,
                 filters=64,
                 lrelu_alpha=0.2,
                 pad_type="reflect",
                 norm_type="batch"):
        super(StridedConv, self).__init__(name="StridedConv")
        self.filters = filters
        self.lrelu_alpha = lrelu_alpha
        self.pad_type = pad_type
        self.norm_type = norm_type

        if self.pad_type == "reflect":
            self.pad1 = ReflectionPadding2D()
        elif self.pad_type == "constant":
            self.pad1 = ZeroPadding2D()

        self.conv1 = Conv2D(self.filters, 3, strides=(2, 2))
        self.lrelu1 = LeakyReLU(self.lrelu_alpha)

        if self.pad_type == "reflect":
            self.pad2 = ReflectionPadding2D()
        elif self.pad_type == "constant":
            self.pad2 = ZeroPadding2D()

        self.conv2 = Conv2D(self.filters * 2, 3)

        if self.norm_type == "instance":
            self.norm = InstanceNormalization()
        elif self.norm_type == "batch":
            self.norm = BatchNormalization()

        self.lrelu2 = LeakyReLU(self.lrelu_alpha)

    def build(self, input_shape):
        super(StridedConv, self).build(input_shape)

    def call(self, x, training=False):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm(x, training=training)
        x = self.lrelu2(x)
        return x

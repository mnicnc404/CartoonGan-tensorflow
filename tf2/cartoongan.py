import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
    BatchNormalization, ReLU, ZeroPadding2D, Add
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


class Conv(Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 padding=(1, 1),
                 norm_type="instance",
                 pad_type="reflect"):
        super(Conv, self).__init__(name="Conv")
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
        super(Conv, self).build(input_shape)

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

        self.deconv1 = Conv2DTranspose(self.filters,
                                       self.kernel_size,
                                       strides=2)
        self.deconv2 = Conv2DTranspose(self.filters,
                                       self.kernel_size)

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

        x = self.deconv1(x)
        x = self.deconv2(x)

        boxes = [[0, 0, 1, 1]] * x.shape[0]
        box_indices = list(range(x.shape[0]))
        x = tf.image.crop_and_resize(x, boxes, box_indices, (cur_h, cur_w))

        x = self.norm(x, training=training)
        x = self.act(x)
        return x


class Generator(Model):
    def __init__(self,
                 norm_type="instance",
                 pad_type="reflect",
                 base_filters=64,
                 num_resblocks=8):
        super(Generator, self).__init__(name="Generator")
        self.norm_type = norm_type
        self.pad_type = pad_type
        self.base_filters = base_filters
        self.num_resblocks = num_resblocks
        self.flat_conv1 = Conv(filters=self.base_filters,
                               kernel_size=7,
                               padding=(3, 3),
                               norm_type=self.norm_type,
                               pad_type=self.pad_type)
        self.down_conv1 = DownSampleConv(filters=self.base_filters * 2,
                                         kernel_size=3,
                                         norm_type=self.norm_type,
                                         pad_type=self.pad_type)
        self.down_conv2 = DownSampleConv(filters=self.base_filters * 4,
                                         kernel_size=3,
                                         norm_type=self.norm_type,
                                         pad_type=self.pad_type)
        self.residual_blocks = [
            ResBlock(self.base_filters * 4, 3)
            for _ in range(self.num_resblocks)]

        self.up_conv1 = UpSampleConv(filters=self.base_filters * 2,
                                     kernel_size=3,
                                     norm_type=self.norm_type)
        self.up_conv2 = UpSampleConv(filters=self.base_filters,
                                     kernel_size=3,
                                     norm_type=self.norm_type)

        if self.pad_type == "reflect":
            self.final_pad = ReflectionPadding2D((3, 3))
        elif self.pad_type == "constant":
            self.final_pad = ZeroPadding2D(3)

        self.final_conv = Conv2D(3, 7)

    def build(self, input_shape):
        super(Generator, self).build(input_shape)

    def call(self, x, training=False):
        x = self.flat_conv1(x, training=training)
        x = self.down_conv1(x, training=training)
        x = self.down_conv2(x, training=training)
        for block in self.residual_blocks:
            x = block(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.final_pad(x)
        x = self.final_conv(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == "__main__":
    import numpy as np
    f = 3
    k = 3
    s = (1, 64, 64, 3)
    nx = np.random.rand(*s)
    nx = nx.astype(np.float32)

    custom_layers = [
        Conv(f, k),
        DownSampleConv(f, k),
        ResBlock(f, k),
        UpSampleConv(f, k)
    ]

    for layer in custom_layers:
        tf.keras.backend.clear_session()
        layer.build(s)
        layer.summary()
        print()
        print()

    tf.keras.backend.clear_session()
    g = Generator()
    shape = (1, 256, 256, 3)
    nx = np.random.rand(*shape).astype(np.float32)
    t = tf.keras.Input(shape=nx.shape[1:], batch_size=nx.shape[0])
    out = g(t)
    g.summary()
    print(f"Input  shape: {nx.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == shape, "Output shape doesn't match input shape"
    print("Generator's output shape is exactly the same as shape of input.")


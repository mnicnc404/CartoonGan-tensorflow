import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from layers import ReflectionPadding2D, FlatConv, DownSampleConv,\
    ResBlock, UpSampleConv


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
        self.flat_conv1 = FlatConv(filters=self.base_filters,
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
    nx = np.random.rand(*s).astype(np.float32)

    custom_layers = [
        FlatConv(f, k),
        DownSampleConv(f, k),
        ResBlock(f, k),
        UpSampleConv(f, k)
    ]

    for layer in custom_layers:
        tf.keras.backend.clear_session()
        out = layer(nx)
        layer.summary()
        print(f"Input  Shape: {nx.shape}")
        print(f"Output Shape: {out.shape}")
        print("\n" * 2)

    tf.keras.backend.clear_session()
    g = Generator()
    shape = (1, 256, 256, 3)
    nx = np.random.rand(*shape).astype(np.float32)
    t = tf.keras.Input(shape=nx.shape[1:], batch_size=nx.shape[0])
    out = g(t)
    g.summary()
    print(f"Input  Shape: {nx.shape}")
    print(f"Output Shape: {out.shape}")
    assert out.shape == shape, "Output shape doesn't match input shape"
    print("Generator's output shape is exactly the same as shape of input.")


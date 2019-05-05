import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation
from layers import FlatConv, ConvBlock, ResBlock, UpSampleConv
from layers import get_padding, DownShuffleUnitV2, BasicShuffleUnitV2


class Generator(Model):
    def __init__(self,
                 norm_type="instance",
                 pad_type="constant",
                 base_filters=64,
                 num_resblocks=8,
                 light=False):
        super(Generator, self).__init__(name="Generator")
        if light:
            downconv = DownShuffleUnitV2
            resblock = BasicShuffleUnitV2
            base_filters += 32
            end_ksize = 5
        else:
            downconv = ConvBlock
            resblock = ResBlock
            end_ksize = 7
        upconv = UpSampleConv
        self.flat_conv1 = FlatConv(filters=base_filters,
                                   kernel_size=end_ksize,
                                   norm_type=norm_type,
                                   pad_type=pad_type)
        self.down_conv1 = downconv(mid_filters=base_filters,
                                   filters=base_filters * 2,
                                   kernel_size=3,
                                   stride=2,
                                   norm_type=norm_type,
                                   pad_type=pad_type)
        self.down_conv2 = downconv(mid_filters=base_filters,
                                   filters=base_filters * 4,
                                   kernel_size=3,
                                   stride=2,
                                   norm_type=norm_type,
                                   pad_type=pad_type)
        self.residual_blocks = tf.keras.models.Sequential([
            resblock(
                filters=base_filters * 4,
                kernel_size=3) for _ in range(num_resblocks)])
        self.up_conv1 = upconv(filters=base_filters * 2,
                               kernel_size=3,
                               norm_type=norm_type,
                               pad_type=pad_type,
                               light=light)
        self.up_conv2 = upconv(filters=base_filters,
                               kernel_size=3,
                               norm_type=norm_type,
                               pad_type=pad_type,
                               light=light)

        end_padding = (end_ksize - 1) // 2
        end_padding = (end_padding, end_padding)
        self.final_conv = tf.keras.models.Sequential([
            get_padding(pad_type, end_padding),
            Conv2D(3, end_ksize)])
        self.final_act = Activation("tanh")

    def build(self, input_shape):
        super(Generator, self).build(input_shape)

    def call(self, x, training=False):
        x = self.flat_conv1(x, training=training)
        x = self.down_conv1(x, training=training)
        x = self.down_conv2(x, training=training)
        x = self.residual_blocks(x, training=training)
        x = self.up_conv1(x, training=training)
        x = self.up_conv2(x, training=training)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


if __name__ == "__main__":
    import numpy as np
    f = 3
    k = 3
    s = (1, 64, 64, 3)
    nx = np.random.rand(*s).astype(np.float32)

    custom_layers = [
        FlatConv(f, k),
        ConvBlock(f, k),
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
    out = g(t, training=False)
    g.summary()
    print(f"Input  Shape: {nx.shape}")
    print(f"Output Shape: {out.shape}")
    assert out.shape == shape, "Output shape doesn't match input shape"
    print("Generator's output shape is exactly the same as shape of input.")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from keras_contrib.layers import InstanceNormalization
from layers import ZeroPadding2D, ReflectionPadding2D, StridedConv


class Discriminator(Model):
    def __init__(self,
                 base_filters=32,
                 lrelu_alpha=0.2,
                 pad_type="reflect",
                 norm_type="batch"):
        super(Discriminator, self).__init__(name="Discriminator")
        if pad_type == "reflect":
            self.flat_pad = ReflectionPadding2D()
        elif pad_type == "constant":
            self.flat_pad = ZeroPadding2D()
        else:
            raise ValueError(f"pad_type not recognized {pad_type}")

        self.flat_conv = Conv2D(base_filters, 3)
        self.flat_lru = LeakyReLU(lrelu_alpha)
        self.strided_conv1 = StridedConv(base_filters * 2,
                                         lrelu_alpha,
                                         pad_type,
                                         norm_type)
        self.strided_conv2 = StridedConv(base_filters * 4,
                                         lrelu_alpha,
                                         pad_type,
                                         norm_type)
        self.conv2 = Conv2D(base_filters * 8, 3)

        if norm_type == "instance":
            self.norm = InstanceNormalization()
        elif norm_type == "batch":
            self.norm = BatchNormalization()

        self.lrelu = LeakyReLU(lrelu_alpha)

        self.final_conv = Conv2D(1, 3)

    def build(self, input_shape):
        super(Discriminator, self).build(input_shape)

    def call(self, x, training=False):
        x = self.flat_pad(x)
        x = self.flat_conv(x)
        x = self.flat_lru(x)
        x = self.strided_conv1(x, training=training)
        x = self.strided_conv2(x, training=training)
        x = self.conv2(x)
        x = self.norm(x, training=training)
        x = self.lrelu(x)
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    import numpy as np

    shape = (1, 256, 256, 3)
    nx = np.random.rand(*shape).astype(np.float32)
    t = tf.keras.Input(shape=nx.shape[1:], batch_size=nx.shape[0])
    tf.keras.backend.clear_session()
    sc = StridedConv(t.shape[-1])
    out = sc(t)
    sc.summary()
    print(f"Input  Shape: {t.shape}")
    print(f"Output Shape: {out.shape}")
    print("\n" * 2)

    tf.keras.backend.clear_session()
    d = Discriminator()
    out = d(t)
    d.summary()
    print(f"Input  Shape: {t.shape}")
    print(f"Output Shape: {out.shape}")

import numpy as np
import tensorflow as tf


class FixedVGG():

    def __init__(self, npy_path="cut_fixed_vgg.npy", scope_prefix="vgg"):
        self.scope_prefix = scope_prefix
        params = np.load(npy_path).item()
        keys = params.keys()
        # scaling from [0, 1] to [-1, 1]
        self.MEAN = [m * 2 - 1 for m in [0.485, 0.456, 0.406]]
        self.STD = [s * 2 for s in [0.229, 0.224, 0.225]]
        self._init_constant(keys, params)

    def _init_constant(self, keys, params):
        self.consts = {}
        self.layer_keys = []
        self.consts['vggmean'] = tf.constant([[[self.MEAN]]], tf.float32)
        self.consts['vggstd'] = tf.constant([[[self.STD]]], tf.float32)
        with tf.name_scope(f"{self.scope_prefix}_constants"):
            for key in keys:
                layer_key = key.replace("_weight", "").replace("_bias", "")
                if layer_key not in self.layer_keys:
                    self.layer_keys.append(layer_key)
                self.consts[key] = tf.constant(params[key], name=key)
        self.layer_keys = sorted(self.layer_keys)

    def build_graph(self, x, normalize=True):
        with tf.name_scope(f"{self.scope_prefix}_ops"):
            if normalize:
                x = (x - self.consts['vggmean']) / self.consts['vggstd']
            for key in self.layer_keys:
                x = tf.nn.conv2d(x, self.consts["%s_weight" % key], [1, 1, 1, 1], "SAME")
                x = tf.nn.bias_add(x, self.consts["%s_bias" % key])
                if key != "conv4_4":
                    x = tf.nn.relu(x)
                if key in ("conv1_2", "conv2_2", "conv3_4"):
                    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        return x

    def __call__(self, x, **kwargs):
        return self.build_graph(x, **kwargs)


def _test():
    import os
    v1 = FixedVGG(scope_prefix="vgg1")
    x = tf.placeholder(tf.float32, [None, 256, 256, 3])
    nx1 = ((np.random.rand(2, 256, 256, 3) - 0.5) * 2).astype(np.float32)
    outop1 = v1(x)
    writer = tf.summary.FileWriter(os.path.join("tmp", "bruns"), tf.get_default_graph())
    writer.close()
    with tf.Session() as sess:
        nv1 = sess.run(outop1, {x: nx1})
    print(nv1.shape)


if __name__ == "__main__":
    _test()

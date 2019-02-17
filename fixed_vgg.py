import numpy as np
import tensorflow as tf


class FixedVGG():

    def __init__(self, npy_path="cut_fixed_vgg.npy"):
        params = np.load(npy_path, encoding="latin1").item()
        keys = sorted(params.keys())
        self._init_constant(keys, params)

    def _init_constant(self, keys, params):
        self.consts = {}
        self.layer_keys = []
        with tf.name_scope("vgg_constants"):
            for key in keys:
                self.layer_keys.append(key)
                self.consts["%s_weight" % key] = tf.constant(params[key][0], name="%s_weight" % key)
                self.consts["%s_bias" % key] = tf.constant(params[key][1], name="%s_bias" % key)
                if key == "conv4_4":
                    break

    def build_graph(self, x):
        with tf.name_scope("vgg_ops"):
            for key in self.layer_keys:
                x = tf.nn.conv2d(x, self.consts["%s_weight" % key], [1, 1, 1, 1], "SAME")
                x = tf.nn.bias_add(x, self.consts["%s_bias" % key])
                if key != "conv4_4":
                    x = tf.nn.relu(x)
                if key in ("conv1_2", "conv2_2", "conv3_4"):
                    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        return x


def _test():
    import os
    v = FixedVGG()
    print(v.consts.keys())
    writer = tf.summary.FileWriter(os.path.join("tmp", "gruns"), tf.get_default_graph())
    writer.close()
    x1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    x2 = tf.placeholder(tf.float32, [2, 224, 224, 3])
    nx = np.random.rand(2, 224, 224, 3).astype(np.float32)
    v1 = v.build_graph(x1)
    v2 = v.build_graph(x2)
    writer = tf.summary.FileWriter(os.path.join("tmp", "vruns"), tf.get_default_graph())
    writer.close()
    with tf.Session() as sess:
        nv1 = sess.run(v1, {x1: nx})
        nv2 = sess.run(v2, {x2: nx})
    print(np.sqrt(np.mean((nv1-nv2)**2)))
    try:
        vv = FixedVGG("fixed_vgg.npy")
        x3 = tf.placeholder(tf.float32, [None, 224, 224, 3])
        v3 = vv.build_graph(x3)
        with tf.Session() as sess:
            nv3 = sess.run(v3, {x3: nx})
        print(np.sqrt(np.mean((nv1-nv3)**2)))
    except FileNotFoundError:
        print("Original VGG params not found; skipping this test.")


if __name__ == "__main__":
    _test()

import numpy as np
import tensorflow as tf


class FixedVGG():

    def __init__(self, npy_path="cut_fixed_vgg.npy", prereversed=True, scope_prefix="vgg"):
        self.scope_prefix = scope_prefix
        params = np.load(npy_path, encoding="latin1").item()
        keys = sorted(params.keys())
        self.VGGMEAN = [123.68, 116.779, 103.939]  # RGB
        if prereversed:
            # So this network eats RGB instead of BGR
            params[keys[0]][0] = np.flip(params[keys[0]][0], 2)
        self.prereversed = prereversed
        self._init_constant(keys, params)

    def _init_constant(self, keys, params):
        self.consts = {}
        self.layer_keys = []
        self.consts['vggmean'] = tf.constant([[[self.VGGMEAN]]], tf.float32)
        with tf.name_scope(f"{self.scope_prefix}_constants"):
            for key in keys:
                self.layer_keys.append(key)
                self.consts["%s_weight" % key] = tf.constant(params[key][0], name="%s_weight" % key)
                self.consts["%s_bias" % key] = tf.constant(params[key][1], name="%s_bias" % key)
                if key == "conv4_4":
                    break

    def build_graph(self, x):
        with tf.name_scope(f"{self.scope_prefix}_ops"):
            x = (x + 1) * 127.5 - self.consts['vggmean']
            if not self.prereversed:
                x = tf.reverse(x, [3])  # RGB -> BGR
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
    v2 = FixedVGG(prereversed=False, scope_prefix="vgg2")
    x = tf.placeholder(tf.float32, [None, 256, 256, 3])
    nx = ((np.random.rand(2, 256, 256, 3) - 0.5) * 2).astype(np.float32)
    outop1 = v1(x)
    outop2 = v2(x)
    writer = tf.summary.FileWriter(os.path.join("tmp", "bruns"), tf.get_default_graph())
    writer.close()
    with tf.Session() as sess:
        nv1 = sess.run(outop1, {x: nx})
        nv2 = sess.run(outop2, {x: nx})
    print(np.sqrt(np.mean((nv1-nv2)**2)))
    try:
        vv = FixedVGG("fixed_vgg.npy")
        x3 = tf.placeholder(tf.float32, [None, 256, 256, 3])
        v3 = vv(x3)
        with tf.Session() as sess:
            nv3 = sess.run(v3, {x3: nx})
        print(np.sqrt(np.mean((nv1-nv3)**2)))
    except FileNotFoundError:
        print("Original VGG params not found; skipping this test.")


if __name__ == "__main__":
    _test()

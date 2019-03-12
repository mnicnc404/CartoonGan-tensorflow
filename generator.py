import tensorflow as tf
from net_base import NetBase
from modules import coupled_conv, conv_with_in, instance_norm, conv


class Generator(NetBase):

    def __init__(self, conv_arch, input_size=None,
                 base_chs=64, init_params=None, inf_only=False):
        super(Generator, self).__init__(
                input_size, base_chs, init_params, inf_only)
        self.graph_prefix = "Generator"
        self.conv_arch = conv_arch
        if self.conv_arch == "coupled_conv":
            self.num_conv_params = 6
        elif self.conv_arch == "conv_with_in":
            self.num_conv_params = 3

    def build_graph(self, x, reuse=False):
        with tf.variable_scope(self.graph_prefix, reuse=reuse):
            chs = self.base_chs
            input_shape = tf.shape(x)
            self.logger.debug("initial conv: 3, %d" % chs)
            self.logger.debug(f"input_shape: {input_shape}")
            # Init conv
            x = conv(x, 3, chs, 5, 1, 2, 0, False, self.get_params(0))
            x = tf.nn.relu(instance_norm(
                x, chs, 1, 1e-6,
                *self.get_params(1, 3)))
            prev_chs = chs
            par_pos = 3
            mcnt = 2
            # Downsample
            with tf.variable_scope("Downsample", reuse=reuse):
                for i in range(4):
                    if i % 2 == 0:
                        chs *= 2
                        stride = 2
                    else:
                        stride = 1
                    self.logger.debug("downsample conv: %d, %d" % (prev_chs, chs))
                    if self.conv_arch == "coupled_conv":
                        x = coupled_conv(x, prev_chs, chs, 3, stride, True, mcnt,
                                         self.get_params(par_pos, par_pos + self.num_conv_params))
                    elif self.conv_arch == "conv_with_in":
                        x = conv_with_in(x, prev_chs, chs, 3, stride, True, mcnt,
                                         self.get_params(par_pos, par_pos + self.num_conv_params))
                    prev_chs = chs
                    par_pos += self.num_conv_params
                    mcnt += 1
            # ResBlock
            with tf.variable_scope("ResBlock", reuse=reuse):
                for _ in range(8):
                    x1 = x
                    self.logger.debug("res conv: %d, %d" % (prev_chs, chs))
                    if self.conv_arch == "coupled_conv":
                        x = coupled_conv(x, prev_chs, chs, 3, 1, True, mcnt,
                                         self.get_params(par_pos, par_pos + self.num_conv_params))
                        x = coupled_conv(x, prev_chs, chs, 3, 1, False, mcnt + 1,
                                         self.get_params(
                                             par_pos + self.num_conv_params,
                                             par_pos + self.num_conv_params * 2))
                    elif self.conv_arch == "conv_with_in":
                        x = conv_with_in(x, prev_chs, chs, 3, 1, True, mcnt,
                                         self.get_params(par_pos, par_pos + self.num_conv_params))
                        x = conv_with_in(x, prev_chs, chs, 3, 1, False, mcnt + 1,
                                         self.get_params(
                                             par_pos + self.num_conv_params,
                                             par_pos + self.num_conv_params * 2))

                    x += x1  # no relu as suggested by Sam Gross and Michael Wilber
                    mcnt += 2
                    par_pos += self.num_conv_params * 2
            # Upsample
            with tf.variable_scope("Upsample", reuse=reuse):
                cur_h = input_shape[1] // 4
                cur_w = input_shape[2] // 4
                for i in range(4):
                    if i % 2 == 0:
                        cur_h *= 2
                        cur_w *= 2
                        chs /= 2
                        x = tf.image.resize_bilinear(x, (cur_h, cur_w))
                    self.logger.debug(f"upsample conv: {prev_chs}, {chs}, "
                                      f"height: {cur_h}, width: {cur_w}")
                    if self.conv_arch == "coupled_conv":
                        x = coupled_conv(x, prev_chs, chs, 3, 1, True, mcnt,
                                         self.get_params(par_pos, par_pos + self.num_conv_params))
                    elif self.conv_arch == "conv_with_in":
                        x = conv_with_in(x, prev_chs, chs, 3, 1, True, mcnt,
                                         self.get_params(par_pos, par_pos + self.num_conv_params))
                    par_pos += self.num_conv_params
                    mcnt += 1
                    prev_chs = chs
            # Final conv
            self.logger.debug("final conv: %d, %d" % (prev_chs, chs))
            x = conv(x, prev_chs, 3, 5, 1, 2, mcnt, True, *self.get_params(par_pos, par_pos + 2))
            par_pos += 2
            self.logger.debug("%d param tensors traversed" % par_pos)
        if self.to_save_vars is None:
            self.to_save_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.graph_prefix)
            assert len(self.to_save_vars) == par_pos
        if self.saver is None:
            self.saver = tf.train.Saver(self.to_save_vars)
        self.has_graph = True
        return x


def _test():
    import os
    import numpy as np
    import logging
    logging.basicConfig(level=logging.DEBUG)
    size = None
    shape = [1, size, size, 3]
    x = tf.placeholder(tf.float32, shape, name="input")
    net = Generator(input_size=size)
    # net = Generator(conv_arch="conv_with_in", input_size=size)
    out_op = net(x)
    nx = np.random.rand(1, 225, 150, 3).astype(np.float32)
    writer = tf.summary.FileWriter(os.path.join("tmp", "gruns"), tf.get_default_graph())
    writer.close()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(out_op, {x: nx})
        net.save(sess, "tmp", "lul")
        net.export("tmp", "lul", True, True, sess)
    logging.debug(f'out.shape: {out.shape}')


if __name__ == '__main__':
    _test()

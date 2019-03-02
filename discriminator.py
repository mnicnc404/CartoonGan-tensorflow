import logging
import tensorflow as tf
from net_base import NetBase
from modules import coupled_conv, batch_norm, conv


class Discriminator(NetBase):

    def __init__(self, input_size=224, base_chs=32, init_params=None, inf_only=False):
        super(Discriminator, self).__init__(
                input_size, base_chs, init_params, inf_only)
        self.graph_prefix = "Discriminator"
        self.logger = logging.getLogger(self.graph_prefix)

    def build_graph(self, x, reuse=False):
        with tf.variable_scope(self.graph_prefix, reuse=reuse):
            chs = self.base_chs
            self.logger.debug("initial conv: 3, %d" % chs)
            # Init conv
            x = conv(x, 3, chs, 3, 1, 1, 0, False, self.get_params(0))
            x = tf.nn.leaky_relu(batch_norm(
                x, chs, 1, 1e-5,
                *self.get_params(1, 3)))
            prev_chs = chs
            par_pos = 3
            mcnt = 2
            for i in range(5):
                stride = 2 if i == 0 or i == 2 else 1
                chs = chs if i == 2 or i == 4 else chs * 2
                self.logger.debug("conv: %d, %d" % (prev_chs, chs))
                x = tf.nn.leaky_relu(coupled_conv(x, prev_chs, chs, 3, stride, False, mcnt,
                                                  self.get_params(par_pos, par_pos + 6)))
                prev_chs = chs
                par_pos += 6
                mcnt += 1
            self.logger.debug("final conv: %d, 1" % prev_chs)
            x = conv(x, prev_chs, 1, 3, 1, 1, mcnt, True, *self.get_params(par_pos, par_pos + 2))
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
    logging.basicConfig(level=logging.DEBUG)
    size = 224
    x = tf.placeholder(tf.float32, [2, size, size, 3])
    net = Discriminator(input_size=size)
    nx = np.random.rand(2, size, size, 3).astype(np.float32)
    out_op = net.build_graph(x)
    out_op2 = net.build_graph(x, True)
    writer = tf.summary.FileWriter(os.path.join("tmp", "druns"), tf.get_default_graph())
    writer.close()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(out_op, {x: nx})
        out2 = sess.run(out_op2, {x: nx})
    logging.debug(out.shape)
    logging.debug(out2.shape)
    logging.debug(np.sqrt(np.mean((out-out2)**2)))


if __name__ == '__main__':
    _test()

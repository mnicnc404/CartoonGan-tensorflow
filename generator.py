import tensorflow as tf
from net_base import NetBase
from modules import coupled_conv, batch_norm, conv


class Generator(NetBase):

    def __init__(self, input_size=224, base_chs=64, init_param=None, inf_only=False):
        super(Generator, self).__init__()
        self.graph_prefix = "Generator"
        self.input_size = input_size
        self.base_chs = base_chs
        self.init_param = init_param
        self.inf_only = inf_only
        self.is_training = None if inf_only else tf.placeholder(tf.bool, name="is_training")

    def build_graph(self, x, reuse=False):
        with tf.variable_scope(self.graph_prefix, reuse=False):
            chs = self.base_chs
            x = conv(x, 3, chs, 5, 1, 2, 0, False, self.get_par(0))
            x = tf.nn.relu(batch_norm(
                x, chs, self.is_training, 1, self.inf_only, 1e-5, .9,
                *self.get_par(1, 5)))
            prev_chs = chs
            stride = 2
            par_pos = 5
            # Downsample
            mcnt = 2
            for i in range(2, 6):
                if i % 2 == 0:
                    chs *= 2
                    stride = 2
                else:
                    stride = 1
                x = coupled_conv(x, prev_chs, chs, 3, stride, True, self.is_training, mcnt,
                                 self.inf_only, *self.get_par(par_pos, par_pos+10))
                prev_chs = chs
                par_pos += 10
                mcnt += 1
            # ResBlock
            for i in range(6, 14):
                x1 = x
                x = coupled_conv(x, prev_chs, chs, 3, 1, True, self.is_training, mcnt,
                                 self.inf_only, *self.get_par(par_pos, par_pos+10))
                x = coupled_conv(x, prev_chs, chs, 3, 1, False, self.is_training, mcnt+1,
                                 self.inf_only, *self.get_par(par_pos+10, par_pos+20))
                x = tf.nn.relu(x + x1)
                mcnt += 2
                par_pos += 20
        self.to_save_vars = [
            v for v in tf.global_variables() if v.name.startswith(self.graph_prefix)]
        self.saver = None if reuse else tf.train.Saver(self.to_save_vars)
        self.has_graph = True
        return x


def _test():
    import numpy as np
    x = tf.placeholder(tf.float32, [2, 64, 64, 3])
    net = Generator()
    nx = np.random.rand(2, 64, 64, 3).astype(np.float32)
    out_op = net.build_graph(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(out_op, {x: nx, net.is_training: False})
    print(out.shape)


if __name__ == '__main__':
    _test()

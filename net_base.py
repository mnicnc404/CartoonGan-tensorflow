import os
import pickle
import numpy as np
import tensorflow as tf


class NetBase():

    def __init__(self, input_size=224, base_chs=64, init_param=None, inf_only=False):
        self.has_graph = False
        self.input_size = input_size
        self.base_chs = base_chs
        self.init_param = init_param
        self.inf_only = inf_only

    def build_net(self, x):
        raise NotImplementedError

    def get_par(self, i, j=None):
        if j is None:
            return self.init_param[i] if self.init_param is not None else None
        else:
            return self.init_param[i:j] if self.init_param is not None else [None]

    def save(self, sess, directory, fname):
        assert self.has_graph, "Net graph not constructed!"
        if self.saver is not None:
            if not os.path.isdir(directory):
                os.makedirs(directory)
            if not self.saved_graph:
                tf.train.write_graph(
                    sess.graph.as_graph_def(), directory, "%s.pbtxt" % fname, as_text=True)
                self.saved_graph = True
            self.saver.save(sess, os.path.join(directory, fname))

    def load(self, sess, directory, fname=None):
        if self.saver is not None:
            if fname is not None:
                self.saver.restore(sess, os.path.join(directory, fname))
            else:
                self.saver.restore(sess, tf.train.latest_checkpoint(directory))

    def load_from_numpy(self, sess, params=None, path=None):
        assert self.has_graph, "Net graph not constructed!"
        assert bool(params is not None) != bool(path is not None),\
            "1 and only 1 from args \"params\" and \"path\" should be assigned."
        if path is not None:
            with open(path, "rb") as f:
                params = pickle.load(f)
        # If len(params) < len(self.to_save_vars),
        # the tail of self.to_save_vars remains the same
        for var, param in zip(self.to_save_vars, params):
            sess.run(var.assign(param))

    def save_to_numpy(self, sess, path):
        assert self.has_graph, "Net graph not constructed!"
        params = [np.array(sess.run(v)) for v in self.to_save_vars]
        with open(path, "wb") as f:
            pickle.dump(params, f)

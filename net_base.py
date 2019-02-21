import os
import pickle
import logging
import numpy as np
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph


class NetBase():

    def __init__(self, input_size=224, base_chs=64, init_params=None, inf_only=False):
        self.has_graph = False
        self.saved_graph = False
        self.saver = None
        self.input_size = input_size
        self.base_chs = base_chs
        self.init_params = init_params
        self.inf_only = inf_only
        self.logger = logging.getLogger("NetLogger")
        self.logger.setLevel(logging.DEBUG)

    def build_graph(self, x):
        raise NotImplementedError

    def get_params(self, i, j=None):
        if j is None:
            return self.init_params[i] if self.init_params is not None else None
        else:
            return self.init_params[i:j] if self.init_params is not None else [None] * (j - i)

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

    def export(self, directory, ckptname, optimize=False, save_npy=False):
        # FIXME: we should read image name
        # FIXME: placholder of image tensor is just a temporal solution
        size = self.input_size  # only for code readability
        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, [1, size, size, 3], name="input")
            out = self.build_graph(x)
            with tf.Session() as sess:
                self.load(sess, directory, ckptname)
                if save_npy:
                    npypath = os.path.join(directory, f"{ckptname}.pkl")
                    self.save_to_numpy(sess, npypath)
                    self.logger.info(
                        "Params of list of numpy array format saved to %s" % npypath)
                in_graph_def = tf.get_default_graph().as_graph_def()
                out_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess, in_graph_def, [out.op.name])
            if optimize:
                out_graph_def = TransformGraph(out_graph_def, [x.op.name], [out.op.name],
                                               ["strip_unused_nodes",
                                                # "fuse_convolutions",
                                                "fold_constants(ignore_errors=true)",
                                                "fold_batch_norms",
                                                "fold_old_batch_norms"])
                ckptname = f"optimized_{ckptname}"
            ckptpath = os.path.join(directory, f"{ckptname}.pb")
            with tf.gfile.GFile(ckptpath, 'wb') as f:
                f.write(out_graph_def.SerializeToString())
            self.logger.info("Optimized frozen pb saved to %s" % ckptpath)
            node_name_path = os.path.join(directory, "node_names.txt")
            with open(node_name_path, "w") as f:
                f.write(f"{x.op.name}\n{out.op.name}")

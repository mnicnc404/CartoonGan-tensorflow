import os
import logging
import torch
import numpy as np
import tensorflow as tf
from modules import instance_norm


def test_wrapper(test_fn):
    def wrapped_test_fn(*args, **kwargs):
        name = test_fn.__name__
        logging.info(f"Checking {name}{args} ...")
        with tf.Graph().as_default():
            ok = test_fn(*args, **kwargs)
            if ok:
                logging.info(f"{name} OK")
            else:
                logging.error(f"{name} FAILED!")
        return ok
    return wrapped_test_fn


@test_wrapper
def instance_norm_test():
    """Sanity check for self-implemented instance normalization operation"""
    shape = (1, 64, 64, 3)
    chs = shape[-1]
    eps = 1e-06
    mcnt = np.random.randint(1, 100000)
    x = tf.placeholder(tf.float32, shape)
    nx = np.random.rand(*shape)

    # compare with tf.contrib.layers.instance_norm
    # ref: https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/layers/python/layers/normalization.py
    tf_op = tf.contrib.layers.instance_norm(x, epsilon=eps)
    op = instance_norm(x, chs, mcnt, eps=eps)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out_tf = sess.run(tf_op, {x: nx})
        out = sess.run(op, {x: nx})

    # compare with torch.nn.modules.InstanceNorm2d, shape: (N, C, H, W)
    nx_torch = torch.from_numpy(np.moveaxis(nx, -1, 1))
    out_torch = torch.nn.modules.InstanceNorm2d(chs, eps=eps)(nx_torch)
    out_torch = np.moveaxis(np.asarray(out_torch), 1, -1)

    return np.all((out_tf - out) < 1e-09) and np.all((out_torch - out) < 1e-05)


def main(logging_lvl):
    # setup logging
    original_lvl = logging.getLogger().getEffectiveLevel()
    logging.basicConfig(level=logging_lvl)
    logging.basicConfig(level=original_lvl)

    # add all functions with default arguments for testing
    test_functions = [stuff for name, stuff in globals().items() if callable(stuff) and '_test' in name]
    logging.info(f"Number of tests to run: {len(test_functions)}")
    oks = [f() for f in test_functions]

    all_ok = all(oks)
    if all_ok:
        logging.info("All tests passed.")
    else:
        logging.error("ONE / SOME TEST(s) FAILED!")
    return all_ok


if __name__ == '__main__':
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL}
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lvl", type=str, default="info", choices=list(log_levels.keys()))
    parser.add_argument("--show_tf_cpp_log", action="store_true")
    args = parser.parse_args()
    if not args.show_tf_cpp_log:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(log_levels[args.lvl])

"""
Minimal demonstration of tf1 compatibility
"""
import os
from PIL import Image

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except (ImportError, AttributeError):
    import tensorflow as tf

from generator import Generator
from logger import get_logger


# NOTE: TF warnings are too noisy without this
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def main(m_path, out_dir, light=False, test_out=True):
    logger = get_logger("tf1_export", debug=test_out)
    g = Generator(light=light)
    t = tf.placeholder(tf.string, [])
    x = tf.expand_dims(tf.image.decode_jpeg(tf.read_file(t), channels=3), 0)
    x = (tf.cast(x, tf.float32) / 127.5) - 1
    x = g(x, training=False)
    out = tf.cast((tf.squeeze(x, 0) + 1) * 127.5, tf.uint8)
    in_name, out_name = t.op.name, out.op.name
    try:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            g.load_weights(tf.train.latest_checkpoint(m_path))
            in_graph_def = tf.get_default_graph().as_graph_def()
            out_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, in_graph_def, [out_name])
        tf.reset_default_graph()
        tf.import_graph_def(out_graph_def, name='')
    except ValueError:
        logger.error("Failed to load specified weight.")
        logger.error("If you trained your model with --light, "
                     "consider adding --light when executing this script; otherwise, "
                     "do not add --light when executing this script.")
        exit(1)
    makedirs(out_dir)
    m_cnt = 0
    bpath = 'optimized_graph_light' if light else 'optimized_graph'
    out_path = os.path.join(out_dir, f'{bpath}_{m_cnt:04d}.pb')
    while os.path.exists(out_path):
        m_cnt += 1
        out_path = os.path.join(out_dir, f'{bpath}_{m_cnt:04d}.pb')
    with tf.gfile.GFile(out_path, 'wb') as f:
        f.write(out_graph_def.SerializeToString())
    if test_out:
        with tf.Graph().as_default():
            gd = tf.GraphDef()
            with tf.gfile.GFile(out_path, 'rb') as f:
                gd.ParseFromString(f.read())
            tf.import_graph_def(gd, name='')
            tf.get_default_graph().finalize()
            t = tf.get_default_graph().get_tensor_by_name(f"{in_name}:0")
            out = tf.get_default_graph().get_tensor_by_name(f"{out_name}:0")
            from time import time
            start = time()
            with tf.Session() as sess:
                img = Image.fromarray(sess.run(out, {t: "input_images/temple.jpg"}))
                img.show()
            elapsed = time() - start
            logger.debug(f"{elapsed} sec per img")
    logger.info(f"successfully exported ckpt to {out_path}")
    logger.info(f"input var name: {in_name}:0")
    logger.info(f"output var name: {out_name}:0")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", type=str, default="models")
    parser.add_argument("--out_dir", type=str, default='optimized_pbs')
    parser.add_argument("--light", action='store_true')
    parser.add_argument("--not_test_out", action='store_true')
    args = parser.parse_args()
    main(args.m_path, args.out_dir, args.light, not args.not_test_out)

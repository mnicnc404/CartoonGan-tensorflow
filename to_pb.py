try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError as e:
    raise ImportError(f"{e}: not using tf2?")

import os
import numpy as np
from generator import Generator
from PIL import Image


def to_tensor(x):
    x = np.array(x).astype(np.float32)
    x = (x / 127.5) - 1
    return np.expand_dims(x, 0)


def to_img(x):
    x = ((x + 1) * 127.5).astype(np.uint8).squeeze()
    return Image.fromarray(x)


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def main(m_path, out_dir, light=False, test_out=True):
    g = Generator(light=light)
    t = tf.keras.Input(shape=[None, None, 3], batch_size=1)
    out = g(t, training=False)
    if test_out:
        nx = to_tensor(Image.open("../head.jpg"))
    saver = tf.train.Saver()
    print("input_name:", t.op.name)
    print("output_name:", out.op.name)
    in_name, out_name = t.op.name, out.op.name
    with tf.Session() as sess:
        saver.restore(sess, m_path)
        in_graph_def = tf.get_default_graph().as_graph_def()
        out_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, in_graph_def, [out_name])
    tf.reset_default_graph()
    tf.import_graph_def(out_graph_def, name='')
    with tf.Session() as sess:
        pass
    # out_path = os.path.join(out_dir, 'saved_model.pb')
    # makedirs(os.path.join(out_dir, 'assets'))
    # makedirs(os.path.join(out_dir, 'variables'))
    # with tf.gfile.GFile(out_path, 'wb') as f:
    #     f.write(out_graph_def.SerializeToString())
    # if test_out:
    #     with tf.Graph().as_default():
    #         gd = tf.GraphDef()
    #         with tf.gfile.GFile(out_path, 'rb') as f:
    #             gd.ParseFromString(f.read())
    #         tf.import_graph_def(gd, name='')
    #         tf.get_default_graph().finalize()
    #         t = tf.get_default_graph().get_tensor_by_name(f"{in_name}:0")
    #         out = tf.get_default_graph().get_tensor_by_name(f"{out_name}:0")
    #         with tf.Session() as sess:
    #             img = to_img(sess.run(out, {t: nx}))
    #             img.show()
    if test_out:
        with tf.Graph().as_default():
            with tf.Session() as sess:
                tf.saved_model.load(sess, [], out_dir)
                t = tf.get_default_graph().get_tensor_by_name(f"{in_name}:0")
                out = tf.get_default_graph().get_tensor_by_name(f"{out_name}:0")
                img = to_img(sess.run(out, {t: nx}))
                img.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", type=str, default="face_paprika_models/models/generator")
    parser.add_argument("--out_dir", type=str, default='TestSavedModel')
    parser.add_argument("--light", action='store_true')
    args = parser.parse_args()
    main(args.m_path, args.out_dir, args.light)

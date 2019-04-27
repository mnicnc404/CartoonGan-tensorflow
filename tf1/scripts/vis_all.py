import os
from glob import glob
import tensorflow as tf


def main():
    pbs = sorted(glob("*.pb"))
    for pb in pbs:
        tf.reset_default_graph()
        gd = tf.GraphDef()
        with tf.gfile.GFile(pb, 'rb') as f:
            gd.ParseFromString(f.read())
            tf.import_graph_def(gd, name='')
        tf.get_default_graph().finalize()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(
                os.path.join("runs", pb.replace(".pb", "")), sess.graph)
            writer.close()


if __name__ == "__main__":
    main()

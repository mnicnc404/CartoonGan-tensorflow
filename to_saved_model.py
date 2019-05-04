"""
export to SavedModel
"""
import tensorflow as tf
from generator import Generator


def main(m_path, out_dir, light=False):
    g = Generator(light=light)
    g.load_weights(m_path)
    t = tf.keras.Input(shape=[None, None, 3], batch_size=1)
    out = g(t, training=False)
    g.summary()
    print(out.shape)
    print(g.input.name)
    print(g.output.name)
    tf.saved_model.save(g, args.out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", type=str)
    parser.add_argument("--out_dir", type=str, default='NewSavedModel')
    parser.add_argument("--light", action='store_true')
    args = parser.parse_args()
    main(args.m_path, args.out_dir, args.light)

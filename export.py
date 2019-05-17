import os
from subprocess import Popen
import tensorflow as tf
from generator import Generator
from logger import get_logger


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(40)


def main(m_path, out_dir, light):
    logger = get_logger("export")
    try:
        g = Generator(light=light)
        g.load_weights(tf.train.latest_checkpoint(m_path))
        t = tf.keras.Input(shape=[None, None, 3], batch_size=None)
        g(t, training=False)
        g.summary()
    except ValueError as e:
        logger.error(e)
        logger.error("Failed to load specified weight.")
        logger.error("If you trained your model with --light, "
                     "consider adding --light when executing this script; otherwise, "
                     "do not add --light when executing this script.")
        exit(1)
    m_num = 0
    smd = os.path.join(out_dir, "SavedModel")
    tfmd = os.path.join(out_dir, "tfjs_model")
    if light:
        smd += "Light"
        tfmd += "_light"
    saved_model_dir = f"{smd}_{m_num:04d}"
    tfjs_model_dir = f"{tfmd}_{m_num:04d}"
    while os.path.exists(saved_model_dir):
        m_num += 1
        saved_model_dir = f"{smd}_{m_num:04d}"
        tfjs_model_dir = f"{tfmd}_{m_num:04d}"
    tf.saved_model.save(g, saved_model_dir)
    cmd = ['tensorflowjs_converter', '--input_format', 'tf_saved_model',
           '--output_format', 'tfjs_graph_model', saved_model_dir, tfjs_model_dir]
    logger.info(" ".join(cmd))
    exit_code = Popen(cmd).wait()
    if exit_code == 0:
        logger.info(f"Model converted to {saved_model_dir} and {tfjs_model_dir} successfully")
    else:
        logger.error("tfjs model conversion failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", type=str, default='models')
    parser.add_argument("--out_dir", type=str, default='exported_models')
    parser.add_argument("--light", action='store_true')
    args = parser.parse_args()
    main(args.m_path, args.out_dir, args.light)

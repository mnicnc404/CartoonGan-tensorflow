import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
from generator import Generator
from fixed_vgg import FixedVGG


__no_tqdm__ = False
try:
    from tqdm import tqdm
except (ModuleNotFoundError, ImportError):
    __no_tqdm__ = True


def _tqdm(res, *args, **kwargs):
    return res


class Trainer:
    def __init__(self, dataset_name, source_domain, target_domain, input_size, batch_size, show_progress, logger,
                 **kwargs):
        self.dataset_name = dataset_name
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.input_size = input_size
        self.batch_size = batch_size

        if logger is not None:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.info)
            self.logger.warning("You are using the root logger, which has bad a format.")
            self.logger.warning("Please consider passing a better logger.")

        if not show_progress or __no_tqdm__:
            self.tqdm = _tqdm
        else:
            self.tqdm = tqdm


    def _save_generated_images(self, batch_x, step=None):
        batch_size = batch_x.shape[0]
        fig = plt.figure(figsize=(15, batch_size // 8 / 4 * 9))
        for i in range(batch_size):
            fig.add_subplot(batch_size // 8, 8, i + 1)
            plt.imshow(batch_x[i], cmap='Greys_r')
            plt.axis('off')
        if step is not None:
            plt.savefig(f'runs/image_at_step_{step}.png')


    def pretrain_generator(self, pass_vgg=False, learning_rate=1e-5, num_iterations=1000, **kwargs):
        self.logger.info(f"Building dataset using {self.dataset_name} with domain {self.source_domain}...")
        files = glob(f'./datasets/{self.dataset_name}/train{self.source_domain}/*.*')
        ds = tf.data.Dataset.from_tensor_slices(files)

        def image_processing(filename):
            x = tf.read_file(filename)
            x = tf.image.decode_jpeg(x, channels=3)
            img = tf.image.resize_images(x, [self.input_size, self.input_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1
            return img

        ds = ds.map(image_processing).shuffle(10000).repeat().batch(self.batch_size)
        ds_iter = ds.make_initializable_iterator()
        input_images = ds_iter.get_next()

        self.logger.info("Initializing generator...")
        g = Generator(input_size=self.input_size)
        generated_images = g.build_graph(input_images)

        if pass_vgg:
            self.logger.info("Initializing VGG for computing content loss...")
            vgg = FixedVGG()
            vgg_out = vgg.build_graph(input_images)
            g_vgg_out = vgg.build_graph(generated_images)
            content_loss = tf.reduce_mean(tf.abs(vgg_out - g_vgg_out))
        else:
            content_loss = tf.reduce_mean(tf.abs(input_images - generated_images))

        # setup optimizer to update G's variables
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = opt.minimize(content_loss, var_list=g.to_save_vars)

        self.logger.info("Start training...")
        start = datetime.utcnow()
        batch_losses = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ds_iter.initializer.run()

            # load latest checkpoint
            try:
                g.load(sess, 'runs')
            except ValueError:
                pass

            # generate a batch of real images for monitoring G's performance
            real_batch = sess.run(input_images)

            for step in range(num_iterations):
                _, batch_loss = sess.run([train_op, content_loss])
                batch_losses.append(batch_loss)

                if step % 1 == 0:
                    end = datetime.utcnow()
                    time_use = end - start

                    self.logger.info(f"Step {step}, batch_loss: {batch_loss}, time used: {time_use}")
                    fake_batch = sess.run(generated_images, {input_images: real_batch})
                    g.save(sess, 'runs', 'generator')
                    save_generated_images(np.clip(fake_batch, 0, 1), step=step)


def main(**kwargs):
    t = Trainer(**kwargs)
    t.pretrain_generator(**kwargs)


if __name__ == '__main__':
    import argparse
    import sys
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="realworld2cartoon")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--source_domain", type=str, default="A")
    parser.add_argument("--target_domain", type=str, default="B")
    parser.add_argument("--pass_vgg", action="store_true")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--savedir", type=str, default="ckpts")
    parser.add_argument("--logging_lvl", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"])
    parser.add_argument("--logger_out_file", type=str, default=None)
    parser.add_argument("--not_show_progress_bar", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show_tf_cpp_log", action="store_true")

    args = parser.parse_args()

    if not args.show_tf_cpp_log:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args.show_progress = not args.not_show_progress_bar
    log_lvl = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL}
    args.logger = logging.getLogger("Trainer")
    if args.debug:
        args.logger.setLevel(logging.DEBUG)
    else:
        args.logger.setLevel(log_lvl[args.logging_lvl])
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler(sys.stdout)
    stdhandler.setFormatter(formatter)
    args.logger.addHandler(stdhandler)
    if args.logger_out_file is not None:
        fhandler = logging.StreamHandler(open(args.logger_out_file, "a"))
        fhandler.setFormatter(formatter)
        args.logger.addHandler(fhandler)
    kwargs = vars(args)
    main(**kwargs)

import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
from generator import Generator
from fixed_vgg import FixedVGG
from discriminator import Discriminator


__no_tqdm__ = False
try:
    from tqdm import tqdm
except (ModuleNotFoundError, ImportError):
    __no_tqdm__ = True


def _tqdm(res, *args, **kwargs):
    return res


class Trainer:
    def __init__(
        self,
        dataset_name,
        source_domain,
        target_domain,
        input_size,
        batch_size,
        show_progress,
        logger,
        logdir,
        save_dir,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.input_size = input_size
        self.batch_size = batch_size
        self.logdir = logdir
        self.save_dir = save_dir

        if logger is not None:
            self.logger = logger
        else:
            import logging

            self.logger = logging.getLogger()
            self.logger.setLevel(logging.info)
            self.logger.warning(
                "You are using the root logger, which has bad a format."
            )
            self.logger.warning("Please consider passing a better logger.")

        if not show_progress or __no_tqdm__:
            self.tqdm = _tqdm
        else:
            self.tqdm = tqdm

    def _save_generated_images(self, batch_x, image_fname=None):
        batch_size = batch_x.shape[0]
        fig_width = 15
        num_rows = batch_size // 8 if batch_size >= 8 else 1
        fig_height = num_rows / 4 * 9 if batch_size >= 8 else 9


        # FIXME: test only
        # fig_width = fig_height = 7
        # num_rows = 1

        fig = plt.figure(figsize=(fig_width, fig_height))
        for i in range(batch_size):
            fig.add_subplot(num_rows, 8, i + 1)
            # fig.add_subplot(num_rows, 1, i + 1) # FIXME: test only
            plt.imshow(batch_x[i], cmap="Greys_r")
            plt.axis("off")
        if image_fname is not None:
            result_dir = "generated_images"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            plt.savefig(os.path.join(result_dir, image_fname))
        plt.close(fig)

    def get_dataset(self, dataset_name, domain, _type, batch_size):
        files = glob(os.path.join(
            "datasets", dataset_name, f"{_type}{domain}", "*"))
        self.logger.info(f"{len(files)} domain{domain} images available in {_type}{domain} folder.")

        ds = tf.data.Dataset.from_tensor_slices(files)

        def image_processing(filename):
            x = tf.read_file(filename)
            x = tf.image.decode_jpeg(x, channels=3)
            img = tf.image.resize_images(x, [self.input_size, self.input_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1
            return img

        return ds.map(image_processing).shuffle(10000).repeat().batch(batch_size)


    def pretrain_generator(
        self, pass_vgg, learning_rate, num_iterations, tracking_size, reporting_steps, **kwargs
    ):
        self.logger.info(
            f"Building dataset using {self.dataset_name} with domain {self.source_domain}..."
        )

        ds = self.get_dataset(self.dataset_name, self.source_domain, 'train', self.batch_size)
        ds_iter = ds.make_initializable_iterator()
        input_images = ds_iter.get_next()

        self.logger.info("Initializing generator...")
        g = Generator(input_size=self.input_size)
        generated_images = g(input_images)

        if pass_vgg:
            self.logger.info("Initializing VGG for computing content loss...")
            vgg = FixedVGG()
            vgg_out = vgg.build_graph(input_images)
            g_vgg_out = vgg.build_graph(generated_images)
            content_loss = tf.reduce_mean(tf.abs(vgg_out - g_vgg_out))
        else:
            self.logger.info("Define content loss without passing VGG...")
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
                g.load(sess, self.save_dir)
            except ValueError:
                pass

            # generate a batch of real images for monitoring G's performance
            self.logger.info(f"Pick {tracking_size} input images for tracking generator's performance...")
            real_batches = []
            for _ in range(int(tracking_size / self.batch_size)):
                real_batches.append(sess.run(input_images))

            self._save_generated_images(
                np.clip(np.concatenate(real_batches, axis=0), 0, 1),
                image_fname='original_image.png'
            )

            for step in range(num_iterations):
                _, batch_loss = sess.run([train_op, content_loss])
                batch_losses.append(batch_loss)

                if step and step % reporting_steps == 0:
                    fake_batches = []
                    for real_batch in real_batches:
                        fake_batches.append(sess.run(generated_images, {input_images: real_batch}))

                    g.save(sess, self.save_dir, "pretrain_generator_with_vgg")
                    self._save_generated_images(
                        np.clip(np.concatenate(fake_batches, axis=0), 0, 1),
                        image_fname=f"generated_image_at_step_{step}.png"
                    )

                    self.logger.info(
                        f"Finish step {step} with batch_loss: {batch_loss}, time used: {datetime.utcnow() - start}"
                    )

    def train_gan(self, **kwargs):
        # tf.enable_eager_execution()

        self.logger.info("Building data sets for both source / target domains...")
        ds_a = self.get_dataset(self.dataset_name, self.source_domain, 'train', self.batch_size)
        ds_b = self.get_dataset(self.dataset_name, self.target_domain, 'train', self.batch_size)

        ds_a_iter = ds_a.make_initializable_iterator()
        input_a = ds_a_iter.get_next()

        ds_b_iter = ds_b.make_initializable_iterator()
        input_b = ds_b_iter.get_next()

        self.logger.info("Building generator...")
        g = Generator(input_size=self.input_size)
        generated_b = g(input_a)

        self.logger.info("Building discriminator...")
        d = Discriminator(input_size=self.input_size)
        d_real_out = d.build_graph(input_b)
        d_fake_out = d.build_graph(generated_b, reuse=True)

        self.logger.info("Define content loss using VGG...")
        vgg = FixedVGG()
        v_real_out = vgg.build_graph(input_b)
        v_fake_out = vgg.build_graph(generated_b)
        content_loss = tf.reduce_mean(tf.abs(v_real_out - v_fake_out))

        self.logger.info("Define generator/discriminator losses...")
        d_real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(d_real_out), d_real_out)
        d_fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(d_fake_out), d_fake_out)
        d_loss = d_real_loss + d_fake_loss  # TODO: smooth image loss

        g_adversarial_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(d_fake_out), d_fake_out)
        g_loss = g_adversarial_loss + 10 * content_loss

        self.logger.info("Define optimizers...")
        g_optimizer = tf.train.AdamOptimizer(1e-4)
        g_train_op = g_optimizer.minimize(g_loss, var_list=g.to_save_vars)

        d_optimizer = tf.train.AdamOptimizer(1e-4)
        d_train_op = d_optimizer.minimize(d_loss, var_list=d.to_save_vars)

        start = datetime.utcnow()
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            ds_a_iter.initializer.run()
            ds_b_iter.initializer.run()

            self.logger.info("Initializing generator using pre-trained weights...")
            g.load(sess, self.save_dir)


            # real_a = sess.run(input_a)
            # print(real_a.shape)
            # self._save_generated_images(
            #     np.clip(real_a, 0, 1),
            #     image_fname='real_a.png'
            # )
            #
            # fake_b = sess.run(generated_b, {input_a: real_a})
            #
            # self._save_generated_images(
            #     np.clip(fake_b, 0, 1),
            #     image_fname='fake_b.png'
            # )

            tracking_size = 32
            self.logger.info(f"Pick {tracking_size} input images for tracking generator's performance...")
            real_batches = []

            for _ in range(int(tracking_size / self.batch_size)):
                real_batches.append(sess.run(input_a))

            self._save_generated_images(
                np.clip(np.concatenate(real_batches, axis=0), 0, 1),
                image_fname='original_image.png'
            )

            num_iterations = 1000
            for step in range(num_iterations):

                # update D
                _, d_batch_loss = sess.run([d_train_op, d_loss])

                # update G
                _, g_batch_loss = sess.run([g_train_op, g_loss])

                reporting_steps = 50
                if step and step % reporting_steps == 0:
                    fake_batches = []
                    for real_batch in real_batches:
                        fake_batches.append(sess.run(generated_b, {input_a: real_batch}))

                    # g.save(sess, self.save_dir, "gan")
                    self._save_generated_images(
                        np.clip(np.concatenate(fake_batches, axis=0), 0, 1),
                        image_fname=f"gan_image_at_step_{step}.png"
                    )

                    self.logger.info(
                        f"Finish step {step} with d_batch_loss: {d_batch_loss}, g_batch_loss: {g_batch_loss}, time used: {datetime.utcnow() - start}"
                    )




def main(**kwargs):
    t = Trainer(**kwargs)

    pretrain_kwargs = dict(kwargs)
    for k, v in kwargs.items():
        if 'pretrain_' in k:
            pretrain_kwargs[k.replace('pretrain_', '')] = v
    # t.pretrain_generator(**pretrain_kwargs)

    t.train_gan(**kwargs)





    # import tensorflow as tf
    # import numpy as np
    # import os
    # from generator import Generator
    #
    # size = 256
    # x = tf.placeholder(tf.float32, [1, size, size, 3])
    # g = Generator(input_size=size)
    #
    # out_op = g.build_graph(x)
    # print(out_op.op.name)
    # print(out_op.op)
    #
    # with tf.Session() as sess:
    #     # sess.run(tf.global_variables_initializer())
    #
    #     g.load(sess, 'ckpts', 'pretrain_generator_with_vgg')
    #
    #     in_graph_def = tf.get_default_graph().as_graph_def()
    #     out_graph_def = tf.graph_util.convert_variables_to_constants(
    #         sess, in_graph_def, [out_op.op.name])
    #     with tf.gfile.GFile('models/generator.pb', 'wb') as f:
    #         f.write(out_graph_def.SerializeToString())



if __name__ == "__main__":
    import argparse
    import sys
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="realworld2cartoon")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--source_domain", type=str, default="A")
    parser.add_argument("--target_domain", type=str, default="B")
    parser.add_argument("--pretrain_pass_vgg", action="store_true")
    parser.add_argument("--pretrain_learning_rate", type=float, default=1e-5)
    parser.add_argument("--pretrain_num_iterations", type=int, default=3000)
    parser.add_argument("--pretrain_tracking_size", type=int, default=16)
    parser.add_argument("--pretrain_reporting_steps", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--save_dir", type=str, default="ckpts")
    parser.add_argument(
        "--logging_lvl",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
    )
    parser.add_argument("--logger_out_file", type=str, default=None)
    parser.add_argument("--not_show_progress_bar", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show_tf_cpp_log", action="store_true")

    args = parser.parse_args()

    if not args.show_tf_cpp_log:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    args.show_progress = not args.not_show_progress_bar
    log_lvl = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    args.logger = logging.getLogger("Trainer")
    if args.debug:
        args.logger.setLevel(logging.DEBUG)
    else:
        args.logger.setLevel(log_lvl[args.logging_lvl])
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stdhandler = logging.StreamHandler(sys.stdout)
    stdhandler.setFormatter(formatter)
    args.logger.addHandler(stdhandler)
    if args.logger_out_file is not None:
        fhandler = logging.StreamHandler(open(args.logger_out_file, "a"))
        fhandler.setFormatter(formatter)
        args.logger.addHandler(fhandler)
    kwargs = vars(args)
    main(**kwargs)

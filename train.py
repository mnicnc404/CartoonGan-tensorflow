import os
from datetime import datetime
from glob import glob
import logging
import numpy as np
import tensorflow as tf
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
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
        sample_size,
        num_steps,
        reporting_steps,
        content_lambda,
        generator_lr,
        discriminator_lr,
        show_progress,
        logger_name,
        data_dir,
        logdir,
        result_dir,
        pretrain_model_dir,
        model_dir,
        disable_sampling,
        pass_vgg,
        pretrain_learning_rate,
        pretrain_num_steps,
        pretrain_reporting_steps,
        pretrain_generator_name,
        generator_name,
        **kwargs,
    ):
        self.ascii = os.name == "nt"
        self.dataset_name = dataset_name
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.input_size = input_size
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_steps = num_steps
        self.reporting_steps = reporting_steps
        self.content_lambda = content_lambda
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.data_dir = data_dir
        self.logdir = logdir
        self.result_dir = result_dir
        self.pretrain_model_dir = pretrain_model_dir
        self.model_dir = model_dir
        self.disable_sampling = disable_sampling
        self.pass_vgg = pass_vgg
        self.pretrain_learning_rate = pretrain_learning_rate
        self.pretrain_num_steps = pretrain_num_steps
        self.pretrain_reporting_steps = pretrain_reporting_steps
        self.pretrain_generator_name = pretrain_generator_name
        self.generator_name = generator_name

        self.logger = logging.getLogger(logger_name)

        if not show_progress or __no_tqdm__:
            self.tqdm = _tqdm
        else:
            self.tqdm = tqdm

    def _save_generated_images(
        self, batch_x, image_name=None, num_images_per_row=8
    ):
        batch_size = batch_x.shape[0]
        num_rows = (
            batch_size // num_images_per_row if batch_size >= num_images_per_row else 1
        )
        fig_width = 12
        fig_height = 8
        fig = plt.figure(figsize=(fig_width, fig_height))
        for i in range(batch_size):
            fig.add_subplot(num_rows, num_images_per_row, i + 1)
            plt.imshow(batch_x[i])
            plt.axis("off")
        if image_name is not None:
            directory = self.result_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(os.path.join(directory, image_name))
        plt.close(fig)

    def get_dataset(self, dataset_name, domain, _type, batch_size):
        files = glob(os.path.join(self.data_dir, dataset_name, f"{_type}{domain}", "*"))
        self.logger.info(
            f"Found {len(files)} domain{domain} images in {_type}{domain} folder."
        )

        ds = tf.data.Dataset.from_tensor_slices(files)

        def image_processing(filename):
            x = tf.read_file(filename)
            x = tf.image.decode_jpeg(x, channels=3)
            img = tf.image.resize_images(x, [self.input_size, self.input_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1
            return img

        return ds.map(image_processing).shuffle(10000).repeat().batch(batch_size)

    def pretrain_generator(self):
        self.logger.info(
            f"Pretraining generator with {self.pretrain_num_steps} steps, "
            f"batch size: {self.batch_size}..."
        )
        self.logger.info(
            f"Building {self.dataset_name} with domain {self.source_domain}..."
        )

        ds = self.get_dataset(
            self.dataset_name, self.source_domain, "train", self.batch_size
        )
        ds_iter = ds.make_initializable_iterator()
        input_images = ds_iter.get_next()

        self.logger.info("Initializing generator...")
        g = Generator(input_size=None)
        generated_images = g(input_images)

        if self.pass_vgg:
            self.logger.info("Initializing VGG for computing content loss ",
                             f"with content_lambda = {self.content_lambda}...")
            vgg = FixedVGG()
            input_content = vgg.build_graph(input_images)
            generated_content = vgg.build_graph(generated_images)
            content_loss = self.content_lambda * tf.reduce_mean(tf.abs(input_content - generated_content))
        else:
            self.logger.info("Defining content loss without VGG...")
            content_loss = tf.reduce_mean(tf.abs(input_images - generated_images))

        self.logger.info("Setting up optimizer to update generator's parameters...")
        opt = tf.train.AdamOptimizer(learning_rate=self.pretrain_learning_rate)
        train_op = opt.minimize(content_loss, var_list=g.to_save_vars)

        self.logger.info("Start training...")
        start = datetime.utcnow()
        batch_losses = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ds_iter.initializer.run()

            self.logger.info("Loading previous checkpoints...")
            try:
                g.load(sess, self.pretrain_model_dir, self.pretrain_generator_name)
                self.logger.info(
                    f"Successfully loaded {self.pretrain_generator_name}..."
                )
            except (tf.errors.NotFoundError, ValueError):
                self.logger.info(
                    f"Checkpoint with name `{self.pretrain_generator_name}` is not found, "
                    "starting from scratch..."
                )

            if not self.disable_sampling:
                self.logger.info(
                    f"Sampling {self.sample_size} images for tracking generator's performance..."
                )
                real_batches = [
                    sess.run(input_images) for _ in range(self.sample_size//self.batch_size)
                ]

                self._save_generated_images(
                    (np.clip(np.concatenate(real_batches, axis=0), -1, 1) + 1) / 2,
                    image_name="sample_images.png",
                )
            else:
                self.logger.info("Proceed training without sampling images...")

            self.logger.info("Starting training loop...")
            for step in range(1, self.pretrain_num_steps + 1):
                _, batch_loss = sess.run([train_op, content_loss])
                batch_losses.append(batch_loss)

                if step % self.pretrain_reporting_steps == 0:

                    if not self.disable_sampling:
                        fake_batches = [
                            sess.run(
                                generated_images, {input_images: real_b}
                            ) for real_b in real_batches
                        ]
                        self._save_generated_images(
                            (np.clip(np.concatenate(fake_batches, axis=0), -1, 1) + 1) / 2,
                            image_name=f"generated_images_at_step_{step}.png",
                        )

                    self.logger.info(f"Saving checkpoints for step {step}...")
                    g.save(sess, self.model_dir, self.pretrain_generator_name)
                    self.logger.info(
                        "[Step {}] batch_loss: {:.3f}, {} elapsed".format(
                            step, batch_loss, datetime.utcnow() - start
                        )
                    )

                    with open(os.path.join(self.result_dir, "batch_losses.tsv"), "a") as f:
                        f.write(f"{step}\t{batch_loss}\n")

    def train_gan(self, **kwargs):
        self.logger.info("Starting adversarial training...")
        self.logger.info("Building data sets for both source/target/smooth domains...")
        ds_a = self.get_dataset(
            self.dataset_name, self.source_domain, "train", self.batch_size
        )
        ds_b = self.get_dataset(
            self.dataset_name, self.target_domain, "train", self.batch_size
        )
        ds_b_smooth = self.get_dataset(
            self.dataset_name, f"{self.target_domain}_smooth", "train", self.batch_size
        )

        ds_a_iter = ds_a.make_initializable_iterator()
        ds_b_iter = ds_b.make_initializable_iterator()
        ds_b_smooth_iter = ds_b_smooth.make_initializable_iterator()

        input_a = ds_a_iter.get_next()
        input_b = ds_b_iter.get_next()
        input_b_smooth = ds_b_smooth_iter.get_next()

        self.logger.info("Building generator...")
        g = Generator(input_size=self.input_size)
        generated_b = g(input_a)

        self.logger.info("Building discriminator...")
        d = Discriminator(input_size=self.input_size)
        d_real_out = d.build_graph(input_b)
        d_fake_out = d.build_graph(generated_b, reuse=True)
        d_smooth_out = d.build_graph(input_b_smooth, reuse=True)

        self.logger.info("Defining content loss using VGG...")
        vgg = FixedVGG()
        v_real_out = vgg.build_graph(input_a)
        v_fake_out = vgg.build_graph(generated_b)
        content_loss = tf.reduce_mean(tf.abs(v_real_out - v_fake_out))

        self.logger.info("Defining generator/discriminator losses...")
        d_real_loss = tf.reduce_mean(
            tf.losses.sigmoid_cross_entropy(tf.ones_like(d_real_out), d_real_out)
        )
        d_fake_loss = tf.reduce_mean(
            tf.losses.sigmoid_cross_entropy(tf.zeros_like(d_fake_out), d_fake_out)
        )
        d_smooth_loss = tf.reduce_mean(
            tf.losses.sigmoid_cross_entropy(tf.zeros_like(d_smooth_out), d_smooth_out)
        )
        d_loss = d_real_loss + d_fake_loss + d_smooth_loss

        g_adversarial_loss = tf.reduce_mean(
            tf.losses.sigmoid_cross_entropy(tf.ones_like(d_fake_out), d_fake_out)
        )
        g_loss = g_adversarial_loss + self.content_lambda * content_loss

        self.logger.info("Defining optimizers...")
        g_optimizer = tf.train.AdamOptimizer(self.generator_lr)
        g_train_op = g_optimizer.minimize(g_loss, var_list=g.to_save_vars)

        d_optimizer = tf.train.AdamOptimizer(self.discriminator_lr)
        d_train_op = d_optimizer.minimize(d_loss, var_list=d.to_save_vars)

        start = datetime.utcnow()
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            ds_a_iter.initializer.run()
            ds_b_iter.initializer.run()
            ds_b_smooth_iter.initializer.run()

            self.logger.info("Loading previous checkpoints...")
            try:
                g.load(sess, self.model_dir, self.generator_name)
                self.logger.info(f"Successfully loaded {self.generator_name}...")
            except (tf.errors.NotFoundError, ValueError):
                self.logger.info(
                    "Previous checkpoint not found, using pre-trained weights..."
                )
                try:
                    g.load(sess, self.pretrain_model_dir, self.pretrain_generator_name)
                    self.logger.info(f"Successfully loaded {self.pretrain_generator_name}...")
                except (tf.errors.NotFoundError, ValueError):
                    self.logger.info(f"{self.pretrain_generator_name} not found, training from scratch...")

            if not self.disable_sampling:
                self.logger.info(
                    f"Sampling {self.sample_size} images for tracking generator's performance..."
                )
                real_batches = [sess.run(input_a) for _ in range(self.sample_size//self.batch_size)]
                self._save_generated_images(
                    (np.clip(np.concatenate(real_batches, axis=0), -1, 1) + 1) / 2,
                    image_name="sample_images.png",
                )
            else:
                self.logger.info("Train without sampling images...")

            self.logger.info("Starting training loop...")
            for step in range(1, self.num_steps + 1):

                self.logger.debug(f"[Step {step}] Training discriminator...")
                _, d_batch_loss = sess.run([d_train_op, d_loss])

                self.logger.debug(f"[Step {step}] Training generator...")
                _, g_batch_loss, g_content_loss, g_adv_loss = sess.run(
                    [g_train_op, g_loss, content_loss, g_adversarial_loss]
                )

                if step % self.reporting_steps == 0:
                    self.logger.debug(f"Saving step {step}'s progress...")
                    if not self.disable_sampling:
                        fake_batches = [
                            sess.run(generated_b, {input_a: real_b}) for real_b in real_batches
                        ]
                        self._save_generated_images(
                            (np.clip(np.concatenate(fake_batches, axis=0), -1, 1) + 1) / 2,
                            image_name=f"gan_images_at_step_{step}.png",
                        )
                    g.save(sess, self.model_dir, self.generator_name)
                    time_elapsed = datetime.utcnow() - start
                    res = ("[Step {}] d_loss: {:.2f}, g_loss: {:.2f}, c_loss: {:.2f}, "
                           "adv_loss: {:.2f}, time elapsed: {}")
                    self.logger.info(
                        res.format(
                            step,
                            d_batch_loss,
                            g_batch_loss,
                            g_content_loss,
                            g_adv_loss,
                            time_elapsed,
                        )
                    )

                    with open(os.path.join(self.result_dir, "gan_losses.tsv"), "a") as f:

                        f.write(
                            f"{step}\t{d_batch_loss}\t{g_batch_loss}\t"
                            f"{g_content_loss}\t{g_adv_loss}\t{time_elapsed}\n"
                        )


def main(**kwargs):
    t = Trainer(**kwargs)

    mode = kwargs["mode"]
    if mode == "full":
        t.pretrain_generator()
        t.train_gan()
    elif mode == "pretrain":
        t.pretrain_generator()
    elif mode == "gan":
        t.train_gan()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "pretrain", "gan"])
    parser.add_argument("--dataset_name", type=str, default="realworld2cartoon")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_size", type=int, default=32)
    parser.add_argument("--source_domain", type=str, default="A")
    parser.add_argument("--target_domain", type=str, default="B")
    parser.add_argument("--num_steps", type=int, default=600_000)
    parser.add_argument("--reporting_steps", type=int, default=100)
    parser.add_argument("--content_lambda", type=float, default=0.01)
    parser.add_argument("--generator_lr", type=float, default=1e-4)
    parser.add_argument("--discriminator_lr", type=float, default=4e-4)
    parser.add_argument("--pass_vgg", action="store_true")
    parser.add_argument("--pretrain_learning_rate", type=float, default=1e-5)
    parser.add_argument("--pretrain_num_steps", type=int, default=60_000)
    parser.add_argument("--pretrain_reporting_steps", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--pretrain_model_dir", type=str, default="ckpts")
    parser.add_argument("--model_dir", type=str, default="ckpts")
    parser.add_argument("--disable_sampling", type=bool, default=False)

    parser.add_argument(
        "--pretrain_generator_name", type=str, default="pretrain_generator"
    )
    parser.add_argument("--generator_name", type=str, default="generator")
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
    args.logger_name = "Trainer"
    logger = logging.getLogger(args.logger_name)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_lvl[args.logging_lvl])
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stdhandler = logging.StreamHandler(sys.stdout)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.logger_out_file is not None:
        fhandler = logging.StreamHandler(open(args.logger_out_file, "a"))
        fhandler.setFormatter(formatter)
        args.addHandler(fhandler)
    kwargs = vars(args)
    main(**kwargs)

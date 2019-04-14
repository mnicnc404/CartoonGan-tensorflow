import os
import logging
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
from tensorflow.keras.applications import VGG19
import numpy as np
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        dataset_name,
        source_domain,
        target_domain,
        gan_type,
        epochs,
        input_size,
        batch_size,
        sample_size,
        reporting_steps,
        content_lambda,
        style_lambda,
        g_adv_lambda,
        d_adv_lambda,
        generator_lr,
        discriminator_lr,
        logger_name,
        data_dir,
        log_dir,
        result_dir,
        checkpoint_dir,
        generator_checkpoint_prefix,
        discriminator_checkpoint_prefix,
        pretrain_checkpoint_prefix,
        pretrain_model_dir,
        model_dir,
        disable_sampling,
        ignore_vgg,
        pretrain_learning_rate,
        pretrain_epochs,
        pretrain_saving_epochs,
        pretrain_reporting_steps,
        pretrain_generator_name,
        generator_name,
        discriminator_name,
        **kwargs,
    ):
        self.ascii = os.name == "nt"
        self.dataset_name = dataset_name
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.gan_type = gan_type
        self.epochs = epochs
        self.input_size = input_size
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.reporting_steps = reporting_steps
        self.content_lambda = content_lambda
        self.style_lambda = style_lambda
        self.g_adv_lambda = g_adv_lambda
        self.d_adv_lambda = d_adv_lambda
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.result_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.generator_checkpoint_prefix = generator_checkpoint_prefix
        self.discriminator_checkpoint_prefix = discriminator_checkpoint_prefix
        self.pretrain_checkpoint_prefix = pretrain_checkpoint_prefix
        self.pretrain_model_dir = pretrain_model_dir
        self.model_dir = model_dir
        self.disable_sampling = disable_sampling
        self.ignore_vgg = ignore_vgg
        self.pretrain_learning_rate = pretrain_learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_saving_epochs = pretrain_saving_epochs
        self.pretrain_reporting_steps = pretrain_reporting_steps
        self.pretrain_generator_name = pretrain_generator_name
        self.generator_name = generator_name
        self.discriminator_name = discriminator_name

        self.logger = logging.getLogger(logger_name)

        if not self.ignore_vgg:
            logger.info("Setting up VGG19 for computing content loss...")
            input_shape = (self.input_size, self.input_size, 3)
            vgg19 = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
            self.vgg = tf.keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer("block4_conv4").output)
        else:
            logger.info("VGG19 will not be used. Content loss will simply imply pixel-wise difference.")
            self.vgg = None

        logger.info(f"Setting up objective functions and metrics using {self.gan_type}...")
        self.content_loss_object = tf.keras.losses.MeanAbsoluteError()
        self.generator_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        if self.gan_type == "gan":
            self.discriminator_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif self.gan_type == "lsgan":
            self.discriminator_loss_object = tf.keras.losses.MeanSquaredError()
        else:
            wrong_msg = f"Non-recognized 'gan_type': {self.gan_type}"
            self.logger.critical(wrong_msg)
            raise ValueError(wrong_msg)

        self.g_total_loss_metric = tf.keras.metrics.Mean("g_total_loss", dtype=tf.float32)
        self.g_adv_loss_metric = tf.keras.metrics.Mean("g_adversarial_loss", dtype=tf.float32)
        self.content_loss_metric = tf.keras.metrics.Mean("content_loss", dtype=tf.float32)
        self.d_total_loss_metric = tf.keras.metrics.Mean("d_total_loss", dtype=tf.float32)
        self.d_real_loss_metric = tf.keras.metrics.Mean("d_real_loss", dtype=tf.float32)
        self.d_fake_loss_metric = tf.keras.metrics.Mean("d_fake_loss", dtype=tf.float32)
        self.d_smooth_loss_metric = tf.keras.metrics.Mean("d_smooth_loss", dtype=tf.float32)

        self.metric_and_names = [
            (self.g_total_loss_metric, "g_total_loss"),
            (self.g_adv_loss_metric, "g_adversarial_loss"),
            (self.content_loss_metric, "content_loss"),
            (self.d_total_loss_metric, "d_total_loss"),
            (self.d_real_loss_metric, "d_real_loss"),
            (self.d_fake_loss_metric, "d_fake_loss"),
            (self.d_smooth_loss_metric, "d_smooth_loss"),
        ]

        logger.info("Setting up checkpoint paths...")
        self.pretrain_checkpoint_prefix = os.path.join(
            self.checkpoint_dir, "pretrain", self.pretrain_checkpoint_prefix)
        self.generator_checkpoint_dir = os.path.join(
            self.checkpoint_dir, self.generator_checkpoint_prefix)
        self.generator_checkpoint_prefix = os.path.join(
            self.generator_checkpoint_dir, self.generator_checkpoint_prefix)
        self.discriminator_checkpoint_dir = os.path.join(
            self.checkpoint_dir, self.discriminator_checkpoint_prefix)
        self.discriminator_checkpoint_prefix = os.path.join(
            self.discriminator_checkpoint_dir, self.discriminator_checkpoint_prefix)

    def _save_generated_images(self, batch_x, image_name=None, num_images_per_row=6):
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

    def get_num_images(self, dataset_name, domain, _type):
        files = glob(os.path.join(self.data_dir, dataset_name, f"{_type}{domain}", "*"))
        return len(files)

    def get_dataset(self, dataset_name, domain, _type, batch_size, repeat=False):
        files = glob(os.path.join(self.data_dir, dataset_name, f"{_type}{domain}", "*"))
        num_images = len(files)
        self.logger.info(
            f"Found {num_images} domain{domain} images in {_type}{domain} folder."
        )
        ds = tf.data.Dataset.from_tensor_slices(files)

        def image_processing(filename):
            x = tf.io.read_file(filename)
            x = tf.image.decode_jpeg(x, channels=3)
            x = tf.image.random_crop(x, (self.input_size, self.input_size, 3))
            x = tf.image.resize_image_with_crop_or_pad(x, self.input_size, self.input_size)
            img = tf.cast(x, tf.float32) / 127.5 - 1
            return img

        ds = ds.map(image_processing).shuffle(num_images)
        if repeat:
            ds = ds.repeat()
        return ds.batch(batch_size)

    def get_sample_images(self, dataset):
        sample_image_dir = os.path.join(self.result_dir, "sample_batches")
        if not os.path.exists(sample_image_dir):
            os.makedirs(sample_image_dir)
        batch_files = sorted(glob(os.path.join(sample_image_dir, "sample_batch_*.npy")))
        if not batch_files:
            self.logger.debug("No existing sample images, generating images from dataset...")
            real_batches = list()
            for image_batch in dataset.take(self.sample_size // self.batch_size):
                real_batches.append(image_batch)

            for i, batch in enumerate(real_batches):
                np.save(os.path.join(sample_image_dir, f"sample_batch_{i}.npy"), batch.numpy())

            self._save_generated_images(
                (np.clip(np.concatenate(real_batches, axis=0), -1, 1) + 1) / 2,
                image_name="sample_images.png",
            )
        else:
            self.logger.debug("Existing sample images found, load them directly.")
            real_batches = [np.load(f) for f in batch_files]

        return real_batches

    @tf.function
    def content_loss(self, input_images, generated_images):
        if self.vgg:
            input_content = self.vgg(input_images)
            generated_content = self.vgg(generated_images)
        else:
            input_content = input_images
            generated_content = generated_images

        return self.content_loss_object(input_content, generated_content)

    @tf.function
    def discriminator_loss(self, real_output, fake_output, smooth_output):
        real_loss = self.discriminator_loss_object(tf.ones_like(real_output), real_output)
        fake_loss = self.discriminator_loss_object(tf.zeros_like(fake_output), fake_output)
        smooth_loss = self.discriminator_loss_object(tf.zeros_like(smooth_output), smooth_output)
        total_loss = real_loss + fake_loss + smooth_loss
        return real_loss, fake_loss, smooth_loss, total_loss

    @tf.function
    def generator_adversarial_loss(self, fake_output):
        return self.generator_loss_object(tf.ones_like(fake_output), fake_output)

    @tf.function
    def pretrain_step(self, input_images, generator, optimizer):

        with tf.GradientTape() as tape:
            generated_images = generator(input_images, training=True)
            c_loss = self.content_lambda * self.content_loss(input_images, generated_images)

        gradients = tape.gradient(c_loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

        self.content_loss_metric(c_loss)

    @tf.function
    def train_step(self, source_images, target_images, smooth_images,
                   generator, discriminator, g_optimizer, d_optimizer):

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            real_output = discriminator(target_images, training=True)
            generated_images = generator(source_images, training=True)
            fake_output = discriminator(generated_images, training=True)
            smooth_out = discriminator(smooth_images, training=True)
            d_real_loss, d_fake_loss, d_smooth_loss, d_total_loss = \
                self.discriminator_loss(real_output, fake_output, smooth_out)

            c_loss = self.content_lambda * self.content_loss(source_images, generated_images)
            g_adv_loss = self.g_adv_lambda * self.generator_adversarial_loss(fake_output)
            g_total_loss = c_loss + g_adv_loss

        d_grads = d_tape.gradient(d_total_loss, discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_total_loss, generator.trainable_variables)

        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        self.g_total_loss_metric(g_total_loss)
        self.g_adv_loss_metric(g_adv_loss)
        self.content_loss_metric(c_loss)
        self.d_total_loss_metric(d_total_loss)
        self.d_real_loss_metric(d_real_loss)
        self.d_fake_loss_metric(d_fake_loss)
        self.d_smooth_loss_metric(d_smooth_loss)

    def pretrain_generator(self):
        self.logger.info(f"Starting to pretrain generator with {self.pretrain_epochs} epochs...")
        self.logger.info(
            f"Building `{self.dataset_name}` dataset with domain `{self.source_domain}`..."
        )
        dataset = self.get_dataset(dataset_name=self.dataset_name,
                                   domain=self.source_domain,
                                   _type="train",
                                   batch_size=self.batch_size)
        self.logger.info(f"Initializing generator with "
                         f"batch_size: {self.batch_size}, input_size: {self.input_size}...")
        generator = Generator()
        generator(tf.keras.Input(
            shape=(self.input_size, self.input_size, 3),
            batch_size=self.batch_size))
        generator.summary()

        self.logger.info("Setting up optimizer to update generator's parameters...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.pretrain_learning_rate)

        self.logger.info(f"Try restoring checkpoint: `{self.pretrain_checkpoint_prefix}`...")
        try:
            checkpoint = tf.train.Checkpoint(generator=generator)
            status = checkpoint.restore(tf.train.latest_checkpoint(
                os.path.join(self.checkpoint_dir, "pretrain")))
            status.assert_consumed()

            self.logger.info(f"Previous checkpoints has been restored.")
            trained_epochs = checkpoint.save_counter.numpy()
            epochs = self.pretrain_epochs - trained_epochs
            if epochs <= 0:
                self.logger.info(f"Already trained {trained_epochs} epochs. Set a larger `pretrain_epochs`...")
                return
            else:
                self.logger.info(f"Already trained {trained_epochs} epochs, {epochs} epochs left to be trained...")
        except AssertionError:
            self.logger.info(f"Checkpoint is not found, training from scratch with {self.pretrain_epochs} epochs...")
            trained_epochs = 0
            epochs = self.pretrain_epochs

        if not self.disable_sampling:
            self.logger.info(f"Sampling {self.sample_size} images for monitoring generator's performance onward...")
            real_batches = self.get_sample_images(dataset)
        else:
            self.logger.info("Proceeding pretraining without sample images...")

        self.logger.info("Starting training loop, setting up summary writer to record progress on TensorBoard...")
        progress_bar = tqdm(list(range(epochs)))
        summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "pretrain"))

        num_images = self.get_num_images(self.dataset_name, self.source_domain, "train")
        num_steps_per_epoch = (num_images // self.batch_size) + 1

        for epoch in progress_bar:
            epoch_idx = trained_epochs + epoch + 1
            progress_bar.set_description(f"Epoch {epoch_idx}")

            for step, image_batch in enumerate(dataset, 1):
                self.pretrain_step(image_batch, generator, optimizer)

                if step % self.pretrain_reporting_steps == 0:

                    if not self.disable_sampling:
                        fake_batches = [generator(real_b) for real_b in real_batches]
                        self._save_generated_images(
                            (np.clip(np.concatenate(fake_batches, axis=0), -1, 1) + 1) / 2,
                            image_name=f"pretrain_generated_images_at_epoch_{epoch_idx}_step_{step}.png",
                        )

                    global_step = (epoch_idx - 1) * num_steps_per_epoch + step
                    with summary_writer.as_default():
                        tf.summary.scalar('content_loss',
                                          self.content_loss_metric.result(),
                                          step=global_step)
                    self.content_loss_metric.reset_states()

            if epoch % self.pretrain_saving_epochs == 0:
                self.logger.info(f"Saving checkpoints after epoch {epoch_idx} ended...")
                checkpoint.save(file_prefix=self.pretrain_checkpoint_prefix)

    def train_gan(self):
        self.logger.info(
            f"Starting adversarial training with {self.epochs} epochs, "
            f"batch size: {self.batch_size}..."
        )
        self.logger.info(f"Building `{self.dataset_name}` datasets for source/target/smooth domains...")
        ds_source = self.get_dataset(dataset_name=self.dataset_name,
                                     domain=self.source_domain,
                                     _type="train",
                                     batch_size=self.batch_size)
        ds_target = self.get_dataset(dataset_name=self.dataset_name,
                                     domain=self.target_domain,
                                     _type="train",
                                     batch_size=self.batch_size,
                                     repeat=True)
        ds_smooth = self.get_dataset(dataset_name=self.dataset_name,
                                     domain=f"{self.target_domain}_smooth",
                                     _type="train",
                                     batch_size=self.batch_size,
                                     repeat=True)

        self.logger.info("Setting up optimizer to update generator and discriminator...")
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.generator_lr)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_lr)
        self.logger.info(f"Initializing generator with "
                         f"batch_size: {self.batch_size}, input_size: {self.input_size}...")
        g = Generator()
        g(tf.keras.Input(
            shape=(self.input_size, self.input_size, 3),
            batch_size=self.batch_size))
        g.summary()

        self.logger.info(f"Searching existing checkpoints: `{self.generator_checkpoint_prefix}`...")
        try:
            g_checkpoint = tf.train.Checkpoint(g=g, g_optimizer=g_optimizer)
            g_checkpoint.restore(
                tf.train.latest_checkpoint(self.generator_checkpoint_dir)).assert_existing_objects_matched()
            self.logger.info(f"Previous checkpoints has been restored.")
            trained_epochs = g_checkpoint.save_counter.numpy()
            epochs = self.epochs - trained_epochs
            if epochs <= 0:
                self.logger.info(f"Already trained {trained_epochs} epochs. Set a larger `pretrain_epochs`...")
                return
            else:
                self.logger.info(f"Already trained {trained_epochs} epochs, {epochs} epochs left to be trained...")
        except AssertionError:
            self.logger.info(
                "Previous checkpoints are not found, trying to load checkpoints from pretraining..."
            )

            try:
                g_checkpoint = tf.train.Checkpoint(generator=g)
                g_checkpoint.restore(tf.train.latest_checkpoint(
                    os.path.join(self.checkpoint_dir, "pretrain"))).assert_existing_objects_matched()
                self.logger.info(f"Successfully loaded `{self.pretrain_checkpoint_prefix}`...")
            except AssertionError:
                self.logger.info("specified pretrained checkpoint is not found, training from scratch...")

            trained_epochs = 0
            epochs = self.epochs

        self.logger.info(f"Initializing discriminator with "
                         f"batch_size: {self.batch_size}, input_size: {self.input_size}...")
        d = Discriminator()
        d(tf.keras.Input(
            shape=(self.input_size, self.input_size, 3),
            batch_size=self.batch_size))
        d.summary()

        self.logger.info(f"Searching existing checkpoints: `{self.discriminator_checkpoint_prefix}`...")
        try:
            d_checkpoint = tf.train.Checkpoint(d=d, d_optimizer=d_optimizer)
            d_checkpoint.restore(
                tf.train.latest_checkpoint(self.discriminator_checkpoint_dir)).assert_existing_objects_matched()
            self.logger.info(f"Previous checkpoints has been restored.")
        except AssertionError:
            self.logger.info("specified checkpoint is not found, training from scratch...")

        if not self.disable_sampling:
            self.logger.info(f"Sampling {self.sample_size} images for monitoring generator's performance onward...")
            real_batches = self.get_sample_images(ds_source)
        else:
            self.logger.info("Proceeding training without sample images...")
        self.logger.info("Setting up summary writer to record progress on TensorBoard...")
        progress_bar = tqdm(range(epochs))
        summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir))

        self.logger.info("Starting training loop...")
        num_images = self.get_num_images(self.dataset_name, self.source_domain, "train")
        num_steps_per_epoch = (num_images // self.batch_size) + 1

        self.logger.info(f"Number of trained epochs: {trained_epochs}, epochs to be trained: {epochs}, "
                         f"batch size: {self.batch_size}")
        for epoch in progress_bar:
            epoch_idx = trained_epochs + epoch + 1
            progress_bar.set_description(f"Epoch {epoch_idx}")

            for step, (source_images, target_images, smooth_images) in enumerate(
                    zip(ds_source, ds_target, ds_smooth), 1):

                self.train_step(source_images, target_images, smooth_images,
                                g, d, g_optimizer, d_optimizer)

                if step % self.reporting_steps == 0:

                    if not self.disable_sampling:
                        fake_batches = [g(real_b) for real_b in real_batches]
                        self._save_generated_images(
                            (np.clip(np.concatenate(fake_batches, axis=0), -1, 1) + 1) / 2,
                            image_name=f"generated_images_at_epoch_{epoch_idx}_step_{step}.png",
                        )

                    global_step = (epoch_idx - 1) * num_steps_per_epoch + step
                    with summary_writer.as_default():
                        for metric, name in self.metric_and_names:
                            tf.summary.scalar(name, metric.result(), step=global_step)
                            metric.reset_states()

                    logger.debug(f"Epoch {epoch_idx}, Step {step} finished, "
                                 f"{global_step * self.batch_size} images processed.")

            self.logger.info(f"Saving checkpoints after epoch {epoch_idx} ended...")
            g_checkpoint.save(file_prefix=self.generator_checkpoint_prefix)
            d_checkpoint.save(file_prefix=self.discriminator_checkpoint_prefix)

            g.save_weights(os.path.join(self.model_dir, "generator"))


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
    parser.add_argument("--sample_size", type=int, default=24)
    parser.add_argument("--source_domain", type=str, default="A")
    parser.add_argument("--target_domain", type=str, default="B")
    parser.add_argument("--gan_type", type=str, default="lsgan", choices=["gan", "lsgan"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--reporting_steps", type=int, default=10)
    parser.add_argument("--content_lambda", type=float, default=10)
    parser.add_argument("--style_lambda", type=float, default=1.)
    parser.add_argument("--g_adv_lambda", type=float, default=1)
    parser.add_argument("--d_adv_lambda", type=float, default=1)
    parser.add_argument("--generator_lr", type=float, default=1e-4)
    parser.add_argument("--discriminator_lr", type=float, default=4e-4)
    parser.add_argument("--ignore_vgg", action="store_true")
    parser.add_argument("--pretrain_learning_rate", type=float, default=1e-5)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--pretrain_saving_epochs", type=int, default=1)
    parser.add_argument("--pretrain_reporting_steps", type=int, default=50)
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--checkpoint_dir", type=str, default="training_checkpoints")
    parser.add_argument("--generator_checkpoint_prefix", type=str, default="generator")
    parser.add_argument("--discriminator_checkpoint_prefix", type=str, default="discriminator")
    parser.add_argument("--pretrain_checkpoint_prefix", type=str, default="pretrain_generator")
    parser.add_argument("--pretrain_model_dir", type=str, default="models")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--disable_sampling", action="store_true")
    # TODO: rearrange the order of options
    parser.add_argument(
        "--pretrain_generator_name", type=str, default="pretrain_generator"
    )
    parser.add_argument("--generator_name", type=str, default="generator")
    parser.add_argument("--discriminator_name", type=str, default="discriminator")
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
    logger.propagate = False
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

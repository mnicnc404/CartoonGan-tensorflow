import os
import gc
from glob import glob
from itertools import product
from random import choice

from imageio import imwrite
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from logger import get_logger
from generator import Generator
from discriminator import Discriminator


@tf.function
def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)


class Trainer:
    def __init__(
        self,
        dataset_name,
        light,
        source_domain,
        target_domain,
        gan_type,
        epochs,
        input_size,
        multi_scale,
        batch_size,
        sample_size,
        reporting_steps,
        content_lambda,
        style_lambda,
        g_adv_lambda,
        d_adv_lambda,
        generator_lr,
        discriminator_lr,
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
        debug,
        **kwargs,
    ):
        self.debug = debug
        self.ascii = os.name == "nt"
        self.dataset_name = dataset_name
        self.light = light
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.gan_type = gan_type
        self.epochs = epochs
        self.input_size = input_size
        self.multi_scale = multi_scale
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.reporting_steps = reporting_steps
        self.content_lambda = float(content_lambda)
        self.style_lambda = float(style_lambda)
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

        self.logger = get_logger("Trainer", debug=debug)
        # NOTE: just minimal demonstration of multi-scale training
        self.sizes = [self.input_size - 32, self.input_size, self.input_size + 32]

        if not self.ignore_vgg:
            self.logger.info("Setting up VGG19 for computing content loss...")
            from tensorflow.keras.applications import VGG19
            from tensorflow.keras.layers import Conv2D
            input_shape = (self.input_size, self.input_size, 3)
            # download model using kwarg weights="imagenet"
            base_model = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
            tmp_vgg_output = base_model.get_layer("block4_conv3").output
            tmp_vgg_output = Conv2D(512, (3, 3), activation='linear', padding='same',
                                    name='block4_conv4')(tmp_vgg_output)
            self.vgg = tf.keras.Model(inputs=base_model.input, outputs=tmp_vgg_output)
            self.vgg.load_weights(os.path.expanduser(os.path.join(
                "~", ".keras", "models",
                "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")), by_name=True)
        else:
            self.logger.info("VGG19 will not be used. "
                             "Content loss will simply imply pixel-wise difference.")
            self.vgg = None

        self.logger.info(f"Setting up objective functions and metrics using {self.gan_type}...")
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.generator_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        if self.gan_type == "gan":
            self.discriminator_loss_object = tf.keras.losses.BinaryCrossentropy(
                from_logits=True)
        elif self.gan_type == "lsgan":
            self.discriminator_loss_object = tf.keras.losses.MeanSquaredError()
        else:
            wrong_msg = f"Non-recognized 'gan_type': {self.gan_type}"
            self.logger.critical(wrong_msg)
            raise ValueError(wrong_msg)

        self.g_total_loss_metric = tf.keras.metrics.Mean("g_total_loss", dtype=tf.float32)
        self.g_adv_loss_metric = tf.keras.metrics.Mean("g_adversarial_loss", dtype=tf.float32)
        if self.content_lambda != 0.:
            self.content_loss_metric = tf.keras.metrics.Mean("content_loss", dtype=tf.float32)
        if self.style_lambda != 0.:
            self.style_loss_metric = tf.keras.metrics.Mean("style_loss", dtype=tf.float32)
        self.d_total_loss_metric = tf.keras.metrics.Mean("d_total_loss", dtype=tf.float32)
        self.d_real_loss_metric = tf.keras.metrics.Mean("d_real_loss", dtype=tf.float32)
        self.d_fake_loss_metric = tf.keras.metrics.Mean("d_fake_loss", dtype=tf.float32)
        self.d_smooth_loss_metric = tf.keras.metrics.Mean("d_smooth_loss", dtype=tf.float32)

        self.metric_and_names = [
            (self.g_total_loss_metric, "g_total_loss"),
            (self.g_adv_loss_metric, "g_adversarial_loss"),
            (self.d_total_loss_metric, "d_total_loss"),
            (self.d_real_loss_metric, "d_real_loss"),
            (self.d_fake_loss_metric, "d_fake_loss"),
            (self.d_smooth_loss_metric, "d_smooth_loss"),
        ]
        if self.content_lambda != 0.:
            self.metric_and_names.append((self.content_loss_metric, "content_loss"))
        if self.style_lambda != 0.:
            self.metric_and_names.append((self.style_loss_metric, "style_loss"))

        self.logger.info("Setting up checkpoint paths...")
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

    def _save_generated_images(self, batch_x, image_name, nrow=2, ncol=4):
        # NOTE: 0 <= batch_x <= 1, float32, numpy.ndarray
        if not isinstance(batch_x, np.ndarray):
            batch_x = batch_x.numpy()
        n, h, w, c = batch_x.shape
        out_arr = np.zeros([h * nrow, w * ncol, 3], dtype=np.uint8)
        for (i, j), k in zip(product(range(nrow), range(ncol)), range(n)):
            out_arr[(h * i):(h * (i+1)), (w * j):(w * (j+1))] = batch_x[k]
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        imwrite(os.path.join(self.result_dir, image_name), out_arr)
        gc.collect()
        return out_arr

    @tf.function
    def random_resize(self, x):
        size = choice(self.sizes)
        return tf.image.resize(x, (size, size))

    @tf.function
    def image_processing(self, filename, is_train=True):
        crop_size = self.input_size
        if self.multi_scale and is_train:
            crop_size += 32
        x = tf.io.read_file(filename)
        x = tf.image.decode_jpeg(x, channels=3)
        if is_train:
            sizes = tf.cast(
                crop_size * tf.random.uniform([2], 0.9, 1.1), tf.int32)
            shape = tf.shape(x)[:2]
            sizes = tf.minimum(sizes, shape)
            x = tf.image.random_crop(x, (sizes[0], sizes[1], 3))
            x = tf.image.random_flip_left_right(x)
        x = tf.image.resize(x, (crop_size, crop_size))
        img = tf.cast(x, tf.float32) / 127.5 - 1
        return img

    def get_dataset(self, dataset_name, domain, _type, batch_size):
        files = glob(os.path.join(self.data_dir, dataset_name, f"{_type}{domain}", "*"))
        num_images = len(files)
        self.logger.info(
            f"Found {num_images} domain{domain} images in {_type}{domain} folder."
        )
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(num_images))

        def fn(fname):
            if self.multi_scale:
                return self.random_resize(self.image_processing(fname, True))
            else:
                return self.image_processing(fname, True)

        ds = ds.apply(tf.data.experimental.map_and_batch(fn, batch_size))
        steps = int(np.ceil(num_images/batch_size))
        # user iter(ds) to avoid generating iterator every epoch
        return iter(ds), steps

    @tf.function
    def pass_to_vgg(self, tensor):
        # NOTE: self.vgg should be fixed
        if self.vgg is not None:
            tensor = self.vgg(tensor)
        return tensor

    @tf.function
    def content_loss(self, input_images, generated_images):
        return self.mae(input_images, generated_images)

    @tf.function
    def style_loss(self, input_images, generated_images):
        input_images = gram(input_images)
        generated_images = gram(generated_images)
        return self.mae(input_images, generated_images)

    @tf.function
    def discriminator_loss(self, real_output, fake_output, smooth_output):
        real_loss = self.discriminator_loss_object(tf.ones_like(real_output), real_output)
        fake_loss = self.discriminator_loss_object(tf.zeros_like(fake_output), fake_output)
        smooth_loss = self.discriminator_loss_object(
            tf.zeros_like(smooth_output), smooth_output)
        total_loss = real_loss + fake_loss + smooth_loss
        return real_loss, fake_loss, smooth_loss, total_loss

    @tf.function
    def generator_adversarial_loss(self, fake_output):
        return self.generator_loss_object(tf.ones_like(fake_output), fake_output)

    @tf.function
    def pretrain_step(self, input_images, generator, optimizer):

        with tf.GradientTape() as tape:
            generated_images = generator(input_images, training=True)
            c_loss = self.content_lambda * self.content_loss(
                self.pass_to_vgg(input_images), self.pass_to_vgg(generated_images))

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

            g_adv_loss = self.g_adv_lambda * self.generator_adversarial_loss(fake_output)
            g_total_loss = g_adv_loss
            # NOTE: self.*_lambdas are fixed
            if self.content_lambda != 0. or self.style_lambda != 0.:
                vgg_generated_images = self.pass_to_vgg(generated_images)
                if self.content_lambda != 0.:
                    c_loss = self.content_lambda * self.content_loss(
                        self.pass_to_vgg(source_images), vgg_generated_images)
                    g_total_loss = g_total_loss + c_loss
                if self.style_lambda != 0.:
                    s_loss = self.style_lambda * self.style_loss(
                        self.pass_to_vgg(target_images[:vgg_generated_images.shape[0]]),
                        vgg_generated_images)
                    g_total_loss = g_total_loss + s_loss

        d_grads = d_tape.gradient(d_total_loss, discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_total_loss, generator.trainable_variables)

        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        self.g_total_loss_metric(g_total_loss)
        self.g_adv_loss_metric(g_adv_loss)
        if self.content_lambda != 0.:
            self.content_loss_metric(c_loss)
        if self.style_lambda != 0.:
            self.style_loss_metric(s_loss)
        self.d_total_loss_metric(d_total_loss)
        self.d_real_loss_metric(d_real_loss)
        self.d_fake_loss_metric(d_fake_loss)
        self.d_smooth_loss_metric(d_smooth_loss)

    def pretrain_generator(self):
        summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, "pretrain"))
        self.logger.info(f"Starting to pretrain generator with {self.pretrain_epochs} epochs...")
        self.logger.info(
            f"Building `{self.dataset_name}` dataset with domain `{self.source_domain}`..."
        )
        dataset, steps_per_epoch = self.get_dataset(dataset_name=self.dataset_name,
                                                    domain=self.source_domain,
                                                    _type="train",
                                                    batch_size=self.batch_size)
        if self.multi_scale:
            self.logger.info(f"Initializing generator with "
                             f"batch_size: {self.batch_size}, input_size: multi-scale...")
        else:
            self.logger.info(f"Initializing generator with "
                             f"batch_size: {self.batch_size}, input_size: {self.input_size}...")
        generator = Generator(base_filters=2 if self.debug else 64, light=self.light)
        generator(tf.keras.Input(
            shape=(self.input_size, self.input_size, 3),
            batch_size=self.batch_size))
        generator.summary()

        self.logger.info("Setting up optimizer to update generator's parameters...")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.pretrain_learning_rate,
            beta_1=0.5)

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
                self.logger.info(f"Already trained {trained_epochs} epochs. "
                                 "Set a larger `pretrain_epochs`...")
                return
            else:
                self.logger.info(f"Already trained {trained_epochs} epochs, "
                                 f"{epochs} epochs left to be trained...")
        except AssertionError:
            self.logger.info(f"Checkpoint is not found, "
                             f"training from scratch with {self.pretrain_epochs} epochs...")
            trained_epochs = 0
            epochs = self.pretrain_epochs

        if not self.disable_sampling:
            val_files = glob(os.path.join(
                self.data_dir, self.dataset_name, f"test{self.source_domain}", "*"))
            val_real_batch = tf.map_fn(
                lambda fname: self.image_processing(fname, False),
                tf.constant(val_files), tf.float32, back_prop=False)
            real_batch = next(dataset)
            while real_batch.shape[0] < self.sample_size:
                real_batch = tf.concat((real_batch, next(dataset)), 0)
            real_batch = real_batch[:self.sample_size]
            with summary_writer.as_default():
                img = np.expand_dims(self._save_generated_images(
                    tf.cast((real_batch + 1) * 127.5, tf.uint8),
                    image_name="pretrain_sample_images.png"), 0,)
                tf.summary.image("pretrain_sample_images", img, step=0)
                img = np.expand_dims(self._save_generated_images(
                    tf.cast((val_real_batch + 1) * 127.5, tf.uint8),
                    image_name="pretrain_val_sample_images.png"), 0,)
                tf.summary.image("pretrain_val_sample_images", img, step=0)
            gc.collect()
        else:
            self.logger.info("Proceeding pretraining without sample images...")

        self.logger.info("Starting pre-training loop, "
                         "setting up summary writer to record progress on TensorBoard...")

        for epoch in range(epochs):
            epoch_idx = trained_epochs + epoch + 1

            for step in tqdm(
                    range(1, steps_per_epoch + 1),
                    desc=f"Pretrain Epoch {epoch + 1}/{epochs}"):
                # NOTE: not following official "for img in dataset" example
                #       since it generates new iterator every epoch and can
                #       hardly be garbage-collected by python
                image_batch = dataset.next()
                self.pretrain_step(image_batch, generator, optimizer)

                if step % self.pretrain_reporting_steps == 0:

                    global_step = (epoch_idx - 1) * steps_per_epoch + step
                    with summary_writer.as_default():
                        tf.summary.scalar('content_loss',
                                          self.content_loss_metric.result(),
                                          step=global_step)
                        if not self.disable_sampling:
                            fake_batch = tf.cast(
                                (generator(real_batch, training=False) + 1) * 127.5, tf.uint8)
                            img = np.expand_dims(self._save_generated_images(
                                    fake_batch,
                                    image_name=(f"pretrain_generated_images_at_epoch_{epoch_idx}"
                                                f"_step_{step}.png")),
                                    0,
                            )
                            tf.summary.image('pretrain_generated_images', img, step=global_step)
                    self.content_loss_metric.reset_states()
            with summary_writer.as_default():
                if not self.disable_sampling:
                    val_fake_batch = tf.cast(
                        (generator(val_real_batch, training=False) + 1) * 127.5, tf.uint8)
                    img = np.expand_dims(self._save_generated_images(
                            val_fake_batch,
                            image_name=("pretrain_val_generated_images_at_epoch_"
                                        f"{epoch_idx}_step_{step}.png")),
                            0,
                    )
                    tf.summary.image('pretrain_val_generated_images', img, step=epoch)

            if epoch % self.pretrain_saving_epochs == 0:
                self.logger.info(f"Saving checkpoints after epoch {epoch_idx} ended...")
                checkpoint.save(file_prefix=self.pretrain_checkpoint_prefix)
            gc.collect()
        del dataset
        gc.collect()

    def train_gan(self):
        self.logger.info("Setting up summary writer to record progress on TensorBoard...")
        summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.logger.info(
            f"Starting adversarial training with {self.epochs} epochs, "
            f"batch size: {self.batch_size}..."
        )
        self.logger.info(f"Building `{self.dataset_name}` "
                         "datasets for source/target/smooth domains...")
        ds_source, steps_per_epoch = self.get_dataset(dataset_name=self.dataset_name,
                                                      domain=self.source_domain,
                                                      _type="train",
                                                      batch_size=self.batch_size)
        ds_target, _ = self.get_dataset(dataset_name=self.dataset_name,
                                        domain=self.target_domain,
                                        _type="train",
                                        batch_size=self.batch_size)
        ds_smooth, _ = self.get_dataset(dataset_name=self.dataset_name,
                                        domain=f"{self.target_domain}_smooth",
                                        _type="train",
                                        batch_size=self.batch_size)
        self.logger.info("Setting up optimizer to update generator and discriminator...")
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.generator_lr, beta_1=.5)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_lr, beta_1=.5)
        if self.multi_scale:
            self.logger.info(f"Initializing generator with "
                             f"batch_size: {self.batch_size}, input_size: multi-scale...")
        else:
            self.logger.info(f"Initializing generator with "
                             f"batch_size: {self.batch_size}, input_size: {self.input_size}...")
        generator = Generator(base_filters=2 if self.debug else 64, light=self.light)
        generator(tf.keras.Input(
            shape=(self.input_size, self.input_size, 3),
            batch_size=self.batch_size))

        self.logger.info(f"Searching existing checkpoints: `{self.generator_checkpoint_prefix}`...")
        try:
            g_checkpoint = tf.train.Checkpoint(generator=generator)
            g_checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.generator_checkpoint_dir)).assert_existing_objects_matched()
            self.logger.info(f"Previous checkpoints has been restored.")
            trained_epochs = g_checkpoint.save_counter.numpy()
            epochs = self.epochs - trained_epochs
            if epochs <= 0:
                self.logger.info(f"Already trained {trained_epochs} epochs. "
                                 "Set a larger `epochs`...")
                return
            else:
                self.logger.info(f"Already trained {trained_epochs} epochs, "
                                 f"{epochs} epochs left to be trained...")
        except AssertionError as e:
            self.logger.warning(e)
            self.logger.warning(
                "Previous checkpoints are not found, trying to load checkpoints from pretraining..."
            )

            try:
                g_checkpoint = tf.train.Checkpoint(generator=generator)
                g_checkpoint.restore(tf.train.latest_checkpoint(
                    os.path.join(
                        self.checkpoint_dir, "pretrain"))).assert_existing_objects_matched()
                self.logger.info("Successfully loaded "
                                 f"`{self.pretrain_checkpoint_prefix}`...")
            except AssertionError:
                self.logger.warning("specified pretrained checkpoint is not found, "
                                    "training from scratch...")

            trained_epochs = 0
            epochs = self.epochs

        if self.multi_scale:
            self.logger.info(f"Initializing discriminator with "
                             f"batch_size: {self.batch_size}, input_size: multi-scale...")
        else:
            self.logger.info(f"Initializing discriminator with "
                             f"batch_size: {self.batch_size}, input_size: {self.input_size}...")
        if self.debug:
            d_base_filters = 2
        elif self.light:
            d_base_filters = 24
        else:
            d_base_filters = 32
        d = Discriminator(base_filters=d_base_filters)
        d(tf.keras.Input(
            shape=(self.input_size, self.input_size, 3),
            batch_size=self.batch_size))

        self.logger.info("Searching existing checkpoints: "
                         f"`{self.discriminator_checkpoint_prefix}`...")
        try:
            d_checkpoint = tf.train.Checkpoint(d=d)
            d_checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.discriminator_checkpoint_dir)).assert_existing_objects_matched()
            self.logger.info(f"Previous checkpoints has been restored.")
        except AssertionError:
            self.logger.info("specified checkpoint is not found, training from scratch...")

        if not self.disable_sampling:
            val_files = glob(os.path.join(
                self.data_dir, self.dataset_name, f"test{self.source_domain}", "*"))
            val_real_batch = tf.map_fn(
                lambda fname: self.image_processing(fname, False),
                tf.constant(val_files), tf.float32, back_prop=False)
            real_batch = next(ds_source)
            while real_batch.shape[0] < self.sample_size:
                real_batch = tf.concat((real_batch, next(ds_source)), 0)
            real_batch = real_batch[:self.sample_size]
            with summary_writer.as_default():
                img = np.expand_dims(self._save_generated_images(
                    tf.cast((real_batch + 1) * 127.5, tf.uint8),
                    image_name="gan_sample_images.png"), 0,)
                tf.summary.image("gan_sample_images", img, step=0)
                img = np.expand_dims(self._save_generated_images(
                    tf.cast((val_real_batch + 1) * 127.5, tf.uint8),
                    image_name="gan_val_sample_images.png"), 0,)
                tf.summary.image("gan_val_sample_images", img, step=0)
            gc.collect()
        else:
            self.logger.info("Proceeding training without sample images...")

        self.logger.info("Starting training loop...")

        self.logger.info(f"Number of trained epochs: {trained_epochs}, "
                         f"epochs to be trained: {epochs}, "
                         f"batch size: {self.batch_size}")
        for epoch in range(epochs):
            epoch_idx = trained_epochs + epoch + 1

            for step in tqdm(
                    range(1, steps_per_epoch + 1),
                    desc=f'Train {epoch + 1}/{epochs}',
                    total=steps_per_epoch):
                source_images, target_images, smooth_images = (
                    ds_source.next(), ds_target.next(), ds_smooth.next())
                self.train_step(source_images, target_images, smooth_images,
                                generator, d, g_optimizer, d_optimizer)

                if step % self.reporting_steps == 0:

                    global_step = (epoch_idx - 1) * steps_per_epoch + step
                    with summary_writer.as_default():
                        for metric, name in self.metric_and_names:
                            tf.summary.scalar(name, metric.result(), step=global_step)
                            metric.reset_states()
                        if not self.disable_sampling:
                            fake_batch = tf.cast(
                                (generator(real_batch, training=False) + 1) * 127.5, tf.uint8)
                            img = np.expand_dims(self._save_generated_images(
                                    fake_batch,
                                    image_name=("gan_generated_images_at_epoch_"
                                                f"{epoch_idx}_step_{step}.png")),
                                    0,
                            )
                            tf.summary.image('gan_generated_images', img, step=global_step)

                    self.logger.debug(f"Epoch {epoch_idx}, Step {step} finished, "
                                      f"{global_step * self.batch_size} images processed.")

            with summary_writer.as_default():
                if not self.disable_sampling:
                    val_fake_batch = tf.cast(
                        (generator(val_real_batch, training=False) + 1) * 127.5, tf.uint8)
                    img = np.expand_dims(self._save_generated_images(
                            val_fake_batch,
                            image_name=("gan_val_generated_images_at_epoch_"
                                        f"{epoch_idx}_step_{step}.png")),
                            0,
                    )
                    tf.summary.image('gan_val_generated_images', img, step=epoch)
            self.logger.info(f"Saving checkpoints after epoch {epoch_idx} ended...")
            g_checkpoint.save(file_prefix=self.generator_checkpoint_prefix)
            d_checkpoint.save(file_prefix=self.discriminator_checkpoint_prefix)

            generator.save_weights(os.path.join(self.model_dir, "generator"))
            gc.collect()
        del ds_source, ds_target, ds_smooth
        gc.collect()


def main(**kwargs):
    t = Trainer(**kwargs)

    mode = kwargs["mode"]
    if mode == "full":
        t.pretrain_generator()
        gc.collect()
        t.train_gan()
    elif mode == "pretrain":
        t.pretrain_generator()
    elif mode == "gan":
        t.train_gan()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "pretrain", "gan"])
    parser.add_argument("--dataset_name", type=str, default="realworld2cartoon")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--source_domain", type=str, default="A")
    parser.add_argument("--target_domain", type=str, default="B")
    parser.add_argument("--gan_type", type=str, default="lsgan", choices=["gan", "lsgan"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--reporting_steps", type=int, default=100)
    parser.add_argument("--content_lambda", type=float, default=10)
    parser.add_argument("--style_lambda", type=float, default=1.)
    parser.add_argument("--g_adv_lambda", type=float, default=1)
    parser.add_argument("--d_adv_lambda", type=float, default=1)
    parser.add_argument("--generator_lr", type=float, default=1e-5)
    parser.add_argument("--discriminator_lr", type=float, default=1e-5)
    parser.add_argument("--ignore_vgg", action="store_true")
    parser.add_argument("--pretrain_learning_rate", type=float, default=1e-5)
    parser.add_argument("--pretrain_epochs", type=int, default=2)
    parser.add_argument("--pretrain_saving_epochs", type=int, default=1)
    parser.add_argument("--pretrain_reporting_steps", type=int, default=100)
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
    parser.add_argument("--not_show_progress_bar", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show_tf_cpp_log", action="store_true")

    args = parser.parse_args()

    if not args.show_tf_cpp_log:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    args.show_progress = not args.not_show_progress_bar
    kwargs = vars(args)
    main(**kwargs)

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


def save_generated_images(batch_x, step=None):
    batch_size = batch_x.shape[0]
    fig = plt.figure(figsize=(15, batch_size // 8 / 4 * 9))
    for i in range(batch_size):
        fig.add_subplot(batch_size // 8, 8, i + 1)
        plt.imshow(batch_x[i], cmap='Greys_r')
        plt.axis('off')
    if step is not None:
        plt.savefig(f'runs/image_at_step_{step}.png')


def pretrain_generator(size=256, batch_size=8, dataset_name='vangogh2photo', domain='A',
                       pass_vgg=False, learning_rate=1e-5, num_iterations=1000):

    # build dataset
    files = glob(f'./datasets/{dataset_name}/train{domain}/*.*')
    ds = tf.data.Dataset.from_tensor_slices(files)

    def image_processing(filename):
        x = tf.read_file(filename)
        x = tf.image.decode_jpeg(x, channels=3)
        img = tf.image.resize_images(x, [size, size])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        return img

    ds = ds.map(image_processing).shuffle(10000).repeat().batch(batch_size)
    ds_iter = ds.make_initializable_iterator()
    input_images = ds_iter.get_next()

    # initialize G
    print("Initializing generator ..")
    g = Generator(input_size=size)
    generated_images = g.build_graph(input_images)

    # real/fake outputs generated through VGG
    if pass_vgg:
        vgg = FixedVGG()
        vgg_out = vgg.build_graph(input_images)
        g_vgg_out = vgg.build_graph(generated_images)
        content_loss = tf.reduce_mean(tf.abs(vgg_out - g_vgg_out))
    else:
        content_loss = tf.reduce_mean(tf.abs(input_images - generated_images))

    # setup optimizer to update G's variables
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(content_loss, var_list=g.to_save_vars)

    # training phase
    print("Start training ..")
    start = datetime.utcnow()
    batch_losses = []
    with tf.Session() as sess:
        ds_iter.initializer.run()

        # load latest checkpoint
        try:
            g.load(sess, 'runs')
        except ValueError:
            sess.run(tf.global_variables_initializer())

        # generate a batch of real images for monitoring G's performance
        real_batch = sess.run(input_images)

        for step in range(num_iterations):
            _, batch_loss = sess.run([train_op, content_loss])
            batch_losses.append(batch_loss)

            if step % 100 == 0:
                end = datetime.utcnow()
                time_use = end - start

                print(f"Step {step}, batch_loss: {batch_loss}, time used: {time_use}")
                fake_batch = sess.run(generated_images, {input_images: real_batch})
                g.save(sess, 'runs', 'generator')
                save_generated_images(np.clip(fake_batch, 0, 1), step=step)


if __name__ == '__main__':
    pretrain_generator(domain='B')
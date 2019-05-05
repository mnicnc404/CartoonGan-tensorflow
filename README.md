# CartoonGAN-TensorFlow2
Generate your own cartoon-style images with [CartoonGAN (CVPR 2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf), powered by [TensorFlow 2.0 Alpha](https://www.tensorflow.org/alpha).

Blog posts with project overview, online demo and gallery of generated anime: 

| Blog post | Language | 
|-----------|----------|
| [Generate Anime using CartoonGAN and TensorFlow 2.0](https://leemeng.tw/generate-anime-using-cartoongan-and-tensorflow2-en.html) | English |
| [用 CartoonGAN 及 TensorFlow 2 生成新海誠與宮崎駿動畫](https://leemeng.tw/generate-anime-using-cartoongan-and-tensorflow2.html) | 中文（Chinese）|

![cat](images/cover.gif)

Top-left part of the gif is the original input image, while all other 3 parts are cartoonized from it using different styles.

This repo demonstrate how you can generate cartoon-style images by:
- visiting the [CartoonGAN web demo](https://leemeng.tw/generate-anime-using-cartoongan-and-tensorflow2-en.html) 
- executing provided [Python script](cartoonize.py) like this:

```bash
python cartoonize.py --styles shinkai hayao hosoda
```

When executed, [cartoonize.py](cartoonize.py) will load pretrained models released by the [author](http://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm) of CartoonGAN and [CartoonGAN-Test-Pytorch-Torch](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch) to turn your original images into cartoon-like images. 

## Cartoonize first, Train later

This repo first focus on how you can **actually use** [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) to cartoonize images. 

For those who want to **train** their own CartoonGANs or to understand how to train a [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network) using latest version of [TensorFlow](https://www.tensorflow.org/), training code implemented completely in [TensorFlow 2.0](https://www.tensorflow.org/alpha) are also provided for reference. See [Train your own models](#train-your-own-models) for more details.

## Getting Started

Basically, there are 3 approachs to generate cartoon-style images in this repo:

| Approach | Description |
| ------------- | ------------- |
| [Cartoonize using TensorFlow.js](#cartoonize-using-tensorflowjs) | Cartoonize images with TensorFlow.js on browser, no setup needed |
| [Cartoonize using Colab Notebook](#cartoonize-using-colab-notebook) | Google Colab let us use free GPUs to cartoonize images faster |
| [Clone this repo and run script](#clone-this-repo-and-run-script) | Suitable for power users and those who want to make this repo better :) |

You can start with preferred approach or watch the demos first (shown below).

## [Cartoonize using TensorFlow.js](https://leemeng.tw/generate-anime-using-cartoongan-and-tensorflow2-en.html)

This is by far the easiest way to interact with the CartoonGAN. Just visit the [web demo](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) and upload your images:

![tfjs-demo](images/tfjs-demo.gif)

You can right-click on the result to save it.

Under the hood, the webpage utilize [TensorFlow.js](https://www.tensorflow.org/js) to load the pretrained models and transform your images. However, due to the computation limits of the browsers, this approach currently only support static and relatively small images. If you want to transform gifs, keep reading.

## [Cartoonize using Colab Notebook](https://colab.research.google.com/drive/1WIZBHix_cYIGsBKa4phIwCq5qXwO8fRX) 

The most exciting thing is to cartoonize existing gifs. We created a [Colab notebook](https://colab.research.google.com/drive/1WIZBHix_cYIGsBKa4phIwCq5qXwO8fRX) which set up everything including [TensorFlow 2.0](https://www.tensorflow.org/alpha) for you to achieve that:

![colab-demo](images/colab-demo.gif)

You got the idea. Try cartoonizing your favorite images using styles available in [the notebook](https://colab.research.google.com/drive/1WIZBHix_cYIGsBKa4phIwCq5qXwO8fRX).

## Clone this repo and run script

There are 2 scenarios that you would want to use this approach:
- you have GPU superior to [Tesla K80](https://www.nvidia.com/en-gb/data-center/tesla-k80/) (default GPU on Colab)
- you're familiar with Python & TensorFlow and want to contribute to this project

### Setup

First clone this repo:

```bash
git clone https://github.com/mnicnc404/CartoonGan-tensorflow.git
```

For environment management, we recommend [Anaconda](https://www.anaconda.com/). Of course, you will definitely need [TensorFlow 2.0 Alpha](https://www.tensorflow.org/alpha):

```bash
conda create -n cartoonize python=3.6 -y
conda activate cartoonize
pip install tensorflow-gpu==2.0.0-alpha0
# use `tensorflow==2.0.0-alpha0` if GPU is not available
```

You will also need TensorFlow version of [keras-contrib](https://github.com/keras-team/keras-contrib) for some custom Keras layers:

```bash
git clone https://www.github.com/keras-team/keras-contrib.git \
    && cd keras-contrib \
    && python convert_to_tf_keras.py \
    && USE_TF_KERAS=1 python setup.py install
```

If all installations complete successfully, you're good to go.

### Example usage

Currently, there are 4 styles available:
- `shinkai`
- `hayao`
- `hosoda`
- `paprika`

For demo purpose, let's assume we want to transform [input_images/temple.jpg](input_images/temple.jpg):

<img src="input_images/temple.jpg" alt="temple" width="33%"/>

To cartoonize this image with `shinkai` and `hayao` styles, you can run:

```commandline
python cartoonize.py \
    --input_dir input_images \
    --output_dir output_images \
    --styles shinkai hayao \
    --comparison_view horizontal
```

![cartoonize-script-demo](images/cartoonize-script-demo.gif)

The transformed result will be saved as [output_images/comparison/temple.jpg](output_images/comparison/temple.jpg) like this:

![transformed_temple.jpg](output_images/comparison/temple.jpg)

The left-most image will be the original image, followed by the styled result specified using `--styles` option.

To explore all options with detailed explaination, simply run `python cartoonize.py -h`:

<img src="images/cartoonize-script-demo.jpg" alt="demo" width="80%"/>

## Train your own models

In this section, we explain how you can train CartoonGAN on your own datasets.

### Setup Environment

Assume you already followed the steps in [previous section](#setup), just install some extra packages for training:

```commandline
pip install -r requirements.txt
```

### Prepare Dataset

You also have to prepare your own dataset and arrange the images under [datasets](datasets) folder as below: 

```text
datasets
└── realworld2cartoon
    ├── testA [751 images]
    ├── testB [363 images]
    ├── trainA [6287 images]
    ├── trainB [3273 images]
    └── trainB_smooth [3273 images]
```    

where `trainA` and `testA` contain real-world images, while `trainB` and `testB` contain cartoon images with desired style. You can run [edge_smooth.py](https://github.com/taki0112/CartoonGAN-Tensorflow/blob/master/edge_smooth.py) to generate `trainB_smooth`.


### Start training

Although you may have to tune hyperparameters to generate best result for your own datasets, you can train with the settings that we found effective in our case as a starting point:

```commandline
python train.py \
    --batch_size 8 \
    --pretrain_epochs 10 \
    --content_lambda .5 \
    --pretrain_learning_rate 1e-4 \
    --g_adv_lambda 4. \
    --generator_lr 8e-5 \
    --discriminator_lr 2e-5 \
    --style_lambda 20.
```

<img src="images/train-demo.gif" alt="train-demo" width="80%"/>

Detailed log messages, model architecture and progress bar are all provided. This enable you to gain a better understanding of what is happening when training a CartoonGAN. 

For further details, we recommend you to read code in [train.py](train.py).

## Gallery of Generated Anime

If you want to view more anime generated by CartoonGAN, please visit the blog article with language you prefer:

| Blog post | Language | 
|-----------|----------|
| [Generate Anime using CartoonGAN and TensorFlow 2.0](https://leemeng.tw/generate-anime-using-cartoongan-and-tensorflow2-en.html#Gallery:-some-anime-we-generated) | English |
| [用 CartoonGAN 及 TensorFlow 2 生成新海誠與宮崎駿動畫](https://leemeng.tw/generate-anime-using-cartoongan-and-tensorflow2.html#%E4%B8%80%E4%BA%9B%E8%BD%89%E6%8F%9B%E5%BE%8C%E7%9A%84%E5%8B%95%E6%BC%AB%E7%B5%90%E6%9E%9C) | 中文（Chinese）|

## Acknowledgement
- Thanks to the author `[Chen et al., CVPR18]` who published this great work
- [CartoonGAN-Test-Pytorch-Torch](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch) where we extracted pretrained Pytorch model weights for TensorFlow usage
- [TensorFlow](https://www.tensorflow.org/) which provide many useful tutorials for learning TensorFlow 2.0:
    - [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/alpha/tutorials/generative/dcgan)
    - [Build a Image Input Pipeline](https://www.tensorflow.org/alpha/tutorials/load_data/images)
    - [Get started with TensorBoard](https://www.tensorflow.org/tensorboard/r2/get_started)
    - [Custom layers](https://www.tensorflow.org/tutorials/eager/custom_layers)
- [Google Colaboratory](https://colab.research.google.com/) which allow us to train the models and [cartoonize images](#cartoonize-using-colab-notebook) using free GPUs
- [TensorFlow.js](https://www.tensorflow.org/js) team which help us a lot when building the [online demo](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) for CartoonGAN

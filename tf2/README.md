# CartoonGAN-Tensorflow2
Generate your own cartoon-style images with [CartoonGAN (CVPR 2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf), powered by [TensorFlow 2.0 Alpha](https://www.tensorflow.org/alpha).

![cat](images/cover.gif)

Top-left part of the gif is the original input image, while all other 3 parts are cartoonized from it using different styles.

This repo demonstrate how you can generate images like this by simply visiting [Generate Anime using CartoonGAN & TensorFlow](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) or using a [Python script](cartoonize.py) written for you. (more on that in a second)

This repo aim to focus on the immediate usage of [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf). For those who want to train their own CartoonGANs or to understand how to train a [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network) using latest version of [TensorFlow](https://www.tensorflow.org/), training code implemented completely in [TensorFlow 2.0](https://www.tensorflow.org/alpha) are also provided for reference. See [Train your own models](#train-your-own-models) section for more details.

## Getting Started

We believe most of you are just interested in cartoonizing your own images (rather than training a CartoonGAN), so here are 3 approachs that can help you do that immediately:

| Approach | Description |
| ------------- | ------------- |
| Visit [this blog](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) and upload images | Cartoonize uploaded images using [TensorFlow.js](https://www.tensorflow.org/js) on browser, no setup needed |
| Execute [this Colab Notebook](#) | Just execute the commands in the notebook, upload images and you're all set|
| Clone this repo and run script | Suitable for power users and who want to make this repo better :) |

## Cartoonize using TensorFlow.js

This is by far the most easiest way to interact with the CartoonGAN. Just [visit the page](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) and upload your images:

[tfjs-demo](tfjs-demo.gif)

Under the hood, TensorFlow.js will load the model and transform your images. However, due to the computation limit of browsers, currently this approach only support static and small images. If you want to generated gif, read next section.

## [TODO] Cartoonize using Colab Notebook 


## [TODO] Clone this repo and run script

When executed, the script will load pretrained models released by the [author](http://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm) of [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) and [CartoonGAN-Test-Pytorch-Torch](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch) to turn your original images into cartoon-like images. 

## [TODO] Train your own models

## [TODO] Gallery

## Acknowledgement
- Thanks to the author `[Chen et al., CVPR18]` who published this great work
- [CartoonGAN-Test-Pytorch-Torch](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch) where we extracted model weights for TensorFlow usage
- [TensorFlow](https://www.tensorflow.org/) which provide many useful tutorials for learning TensorFlow 2.0:
    - [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/alpha/tutorials/generative/dcgan)
    - [Build a Image Input Pipeline](https://www.tensorflow.org/alpha/tutorials/load_data/images)
    - [Get started with TensorBoard](https://www.tensorflow.org/tensorboard/r2/get_started)
    - [Custom layers](https://www.tensorflow.org/tutorials/eager/custom_layers)
- [Google Colab](https://colab.research.google.com/) which allow us to training models using free GPUs
- [TensorFlow.js](https://www.tensorflow.org/js) which we generate [online demo](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) for CartoonGAN without web servers

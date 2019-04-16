# CartoonGAN-TensorFlow2
Generate your own cartoon-style images with [CartoonGAN (CVPR 2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf), powered by [TensorFlow 2.0 Alpha](https://www.tensorflow.org/alpha).

![cat](images/cover.gif)

Top-left part of the gif is the original input image, while all other 3 parts are cartoonized from it using different styles.

This repo demonstrate how you can generate cartoon-style images by:
- visiting the [CartoonGAN web demo](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) 
- using provided [Python script](cartoonize.py) like this:

```bash
python cartoonize.py --styles shinkai hayao hosoda
```

## Main focus of this repo

This repo put a lot of focus on how you can **actually** use [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) to transform images. For those who want to train their own CartoonGANs or to understand how to train a [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network) using latest version of [TensorFlow](https://www.tensorflow.org/), training code implemented completely in [TensorFlow 2.0](https://www.tensorflow.org/alpha) are also provided for reference. See [Train your own models](#train-your-own-models) for more details.

## Getting Started

Basically, there are 3 approachs to generate cartoon-style images:

| Approach | Description |
| ------------- | ------------- |
| 1. Visit [web demo](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) and upload images | Cartoonize images with [TensorFlow.js](https://www.tensorflow.org/js) on browser, no setup needed |
| 2. Execute [this Colab Notebook](#) | Just execute the commands in the notebook, upload images and that's it|
| 3. Clone this repo and run script | Suitable for power users and those who want to make this repo better :) |

## Cartoonize using TensorFlow.js

This is by far the easiest way to interact with the CartoonGAN. Just visit the [web demo](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) and upload your images:

![tfjs-demo](images/tfjs-demo.gif)

Under the hood, we use TensorFlow.js to load the pretrained models and transform your images. However, due to the limitation of browsers, now this approach only support static and small images. If you want to transform gif, proceed to next section.

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
- [Google Colaboratory](https://colab.research.google.com/) which allow us to training models and cartoonize images using free GPUs
- [TensorFlow.js](https://www.tensorflow.org/js) which we generate [online demo](https://leemeng.tw/drafts/generate-anime-using-cartoongan-and-tensorflow.html) for CartoonGAN without web servers

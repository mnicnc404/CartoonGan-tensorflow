# CartoonGAN-Tensorflow2
Cartoonize your images in one command using [CartoonGAN (CVPR 2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf), implemented in [TensorFlow 2.0 Alpha](https://www.tensorflow.org/alpha).

![cat](images/cover.gif)

Top-left of the gif is the original image, while all other 3 parts are cartoonized using different styles (more of that in a second).

This repository allow anyone to generate cartoon-ike images like above using simple [Python script](cartoonize.py). The script load pretrained models released by the [author of the paper](http://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm) and [CartoonGAN-Test-Pytorch-Torch](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch) to perform the cartoonization. 

There is also [TensorFlow 2.0](https://www.tensorflow.org/alpha) implementation of CartoonGAN for those who want to train their own models and understand how to train a GAN using latest version of [TensorFlow](https://www.tensorflow.org/).

## Getting Started


## Acknowledgement
- Thanks to the author `[Chen et al., CVPR18]` who published this work
- [CartoonGAN-Test-Pytorch-Torch](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch) where we extracted model weights for TensorFlow usage
- [TensorFlow](https://www.tensorflow.org/) which provide many useful tutorials:
    - [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/alpha/tutorials/generative/dcgan)
    - [Build a Image Input Pipeline](https://www.tensorflow.org/alpha/tutorials/load_data/images)
    - [Get started with TensorBoard](https://www.tensorflow.org/tensorboard/r2/get_started)
    - [Custom layers](https://www.tensorflow.org/tutorials/eager/custom_layers)
- [Google Colab](https://colab.research.google.com/) which allow me to training models using free GPUs

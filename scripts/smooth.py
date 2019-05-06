"""
credit to https://github.com/taki0112/CartoonGAN-Tensorflow/blob/master/edge_smooth.py
LICENSE for this script: https://github.com/taki0112/CartoonGAN-Tensorflow/blob/master/LICENSE
"""
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


def make_edge_smooth(path):
    file_list = glob(os.path.expanduser(os.path.join(path, 'trainB', '*')))
    save_dir = os.path.expanduser(os.path.join(path, 'trainB_smooth'))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list):
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(bgr_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(
                pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0],
                gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(
                pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1],
                gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(
                pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2],
                gauss))

        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)


def main(path):
    make_edge_smooth(path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help='path to your dataset')
    args = parser.parse_args()
    main(args.path)

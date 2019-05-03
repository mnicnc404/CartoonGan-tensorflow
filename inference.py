import os
import numpy as np
from imageio import imread, imwrite
from generator import Generator


def main(m_path, img_path, out_dir, light=False):
    g = Generator(light=light)
    g.load_weights(m_path)
    img = np.expand_dims(imread(img_path), 0).astype(np.float32) / 127.5 - 1
    out = ((g(img).numpy().squeeze() + 1) * 127.5).astype(np.uint8)
    if out_dir != "" and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if out_dir == "":
        out_dir = "."
    imwrite(os.path.join(out_dir, os.path.split(img_path)[1]), out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", type=str)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--out_dir", type=str, default='out')
    parser.add_argument("--light", action='store_true')
    args = parser.parse_args()
    main(args.m_path, args.img_path, args.out_dir, args.light)

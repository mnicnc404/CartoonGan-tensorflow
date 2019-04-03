import argparse
import numpy as np
from model import build_generator


parser = argparse.ArgumentParser(description="cartoonize real world images to specified cartoon style")
parser.add_argument('-s', '--style', type=str, default='shinkai',
                    choices=["shinkai", "hayao", "hosoda", "paprika", "all"],
                    help="cartoon style to be used")

# TODO: gif

args = parser.parse_args()

print(args)

g = build_generator(style="shinkai")
np.random.seed(9527)
nx = np.random.rand(1, 225, 225, 3).astype(np.float32)
out = g(nx)
tf_out = np.load("tf_out.npy")

diff = np.sqrt(np.mean((out - tf_out) ** 2))
assert diff < 1e-6
import os
import numpy as np
import tensorflow as tf
from generator import Generator

LIGHT = False
KERAS_MODEL_DIR = "models"
WEIGHT_FILE = "light_generator_weights.h5" if LIGHT else "generator_weights.h5"
WEIGHT_PATH = os.path.join(KERAS_MODEL_DIR, WEIGHT_FILE)

MODEL_TYPE = "light" if LIGHT else "normal"
TFJS_MODEL_DIR = os.path.join("tfjs_models", MODEL_TYPE)

shape = (1, 256, 256, 3)
nx = np.random.rand(*shape).astype(np.float32)
t = tf.keras.Input(shape=nx.shape[1:], batch_size=nx.shape[0])

g = Generator(light=LIGHT)
out = g(t)
g.summary()
layer_names = [l.name for l in g.layers]
print(layer_names)

layers_seen = []


for l in g.layers:
    if l.name in layers_seen:
        l._name = l.name + '_1'
    else:
        layers_seen.append(l.name)

layer_names = [l.name for l in g.layers]
print(layer_names)

assert len(set(layer_names)) == len(layer_names), f"Duplicate name in {layer_names}"
g.save_weights(WEIGHT_PATH)


cmd = f"""
    tensorflowjs_converter \
    --input_format=keras \
    {WEIGHT_PATH} \
    {TFJS_MODEL_DIR}
"""
os.system(cmd)

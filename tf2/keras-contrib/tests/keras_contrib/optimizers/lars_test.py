from __future__ import print_function
import numpy as np
from keras_contrib.tests import optimizers
from keras_contrib.optimizers import lars
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def test_base_lars():
    optimizers._test_optimizer(lars(0.01))


def test_nesterov_lars():
    optimizers._test_optimizer(lars(0.01, nesterov=True))

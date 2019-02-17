from glob import glob
import random


class Dataset():

    def __init__(
            self,
            folder_a,
            folder_b,
            size=(224, 224),
            batch_size=64,
            shuffle_buffer=None,
            random_seed=0,
            debug=False):
        random.seed(random_seed)
        self.paths_a = glob(os.path.join(folder_a, "*.jpg"))


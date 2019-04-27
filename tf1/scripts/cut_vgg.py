import os
import numpy as np


def main(path):
    directory, fname = os.path.split(path)
    save_fname = "cut_%s" % fname
    params = np.load(path, encoding="latin1").item()
    keys = sorted(params.keys())
    idx = keys.index("conv4_4")
    for k in keys[idx+1:]:
        del params[k]
    np.save(os.path.join(directory, save_fname), params)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])

import sys
import logging


def get_logger(name, log_file=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler(sys.stdout)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if log_file is not None:
        fhandler = logging.StreamHandler(open(log_file, "a"))
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
    logger.propagate = False
    return logger

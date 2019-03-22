import os
import numpy as np 

import logging
import daiquiri

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

def save_to_npz(x, y, data_dir, filename):
    """
    Save images, labels numpy arrays into a single .npz file.
    :param x: images numpy array
    :param y: labels numpy array
    :param data_dir: directory to store .npz file
    :param filename: file name of the stored .npz file
    """
    logger.debug("CHECK - Dataset size: {}".format(
        (x.shape[0] == y.shape[0]) == True))

    fpath = os.path.join(data_dir, filename)
    if os.path.exists(fpath):
        os.remove(fpath)
        logger.debug("'{}' existed and it has been removed!")
    np.savez(fpath, x=x, y=y)
    logger.info("'{}' saved to '{}'".format(filename, fpath))
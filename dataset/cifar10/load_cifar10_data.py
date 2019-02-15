import os
import numpy as np 
from scipy.io import loadmat

def _get_filesnames(data_dir, split='train'):
    """
    Gets all the cifar10 .mat file names from given directory.
    :param data_dir: given directory of where the source file is stored
    :param split: 'train' or 'test' split
    :return: a list of absolute file paths to source files 
    """
    if split == 'train':
        filenames = [
            os.path.join(data_dir, 'data_batch_%d.mat') % i
            for i in range(1, 6)]
    elif split == 'test':
        filenames = [
            os.path.join(data_dir, 'test_batch.mat')]

    return filenames

def load_cifar10(data_dir, split='train'):
    """
    Implementation of Loading cifar10 dataset from separated .mat files.
    :param data_dir: data direcotry of where cifar10 is stored
    :param split: 'train' or 'test' split 
    :return: images and labels in numpy array format,
             where images - uint8, 0~255, (?, 32, 32, 3),
                   labels - uint8, 0~9, (?, ) # TODO: type unsure
    """
    filenames = _get_filesnames(data_dir, split)
    
    images_list = []
    labels_list = []
    for fn in filenames:
        mat = loadmat(fn)
        images = np.transpose(np.reshape(mat['data'], [-1, 3, 32, 32]), [0, 2, 3, 1])
        labels = np.reshape(mat['labels'], -1)
        images_list.append(images)
        labels_list.append(labels)
    
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return images, labels

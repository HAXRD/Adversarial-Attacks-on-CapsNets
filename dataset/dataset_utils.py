import os 
import numpy as np 
import tensorflow as tf 

import logging
import daiquiri

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

import dataset.mnist.mnist_input as mnist_input
import dataset.cifar10.cifar10_input as cifar10_input

SINGLE_PROCESS = {
    'mnist': mnist_input._single_process,
    'cifar10': cifar10_input._single_process
}

def _feature_process(feature):
    """
    Map function to process batched data inside feature dictionary.
    :param feature: a dictionary contains an image and a label,
                    where image, float32, 0.~1., (h, w, c)
                          label, int32, 0~9, (,)
    :return: a dictionary contains a batch of images and labels
    """
    batched_feature = {
        'images': feature['image'],
        'labels': feature['label']
    }
    return batched_feature

def inputs(dataset_name, total_batch_size, num_gpus, max_epochs, resized_size, 
           data_dir, split):
    """
    A generalized implementation of input pipeline for different datasets.
    :param dataset_name: the name of the dataset
    :param total_batch_size: total number of instances per batch
    :param num_gpus: number of GPUs available for computation distribution
    :param max_epochs: maximum number of epochs to run
    :param resize_size: image size after resizing
    :param data_dir: directory to where the source data is installed
    :param split: split set name after stripping out extension,
                  this argument is not limited to 'train' and 'test',
                  it can be other strings as well.
    :return:
        a dataset object, each instance is a feature dictionary
        a dictionary contains dataset's specifications
    """
    logger.debug("CHECK - npz files: {}".format(
        os.path.exists(os.path.join(data_dir, '{}.npz'.format(split))) == True))

    with np.load(os.path.join(data_dir, '{}.npz'.format(split))) as f:
        x, y = f['x'], f['y']

    specs = {
        'split': split, 
        'total_size': int(x.shape[0]),

        'total_batch_size': int(total_batch_size),
        'steps_per_epoch': int(x.shape[0] // total_batch_size),
        'num_gpus': int(num_gpus),
        'batch_size': int(total_batch_size / num_gpus),
        'max_epochs': int(max_epochs),

        'image_size': x.shape[1],
        'depth': x.shape[3],
        'num_classes': 10
    }

    dataset = tf.data.Dataset.from_tensor_slices((x, y)) # ((h, w, c), (,))
    dataset = dataset.prefetch(
        buffer_size=specs['batch_size']*specs['num_gpus']*2)
    
    if split == 'train':
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
            buffer_size=specs['batch_size']*specs['num_gpus']*10,
            count=specs['max_epochs']))
    else:
        dataset = dataset.repeat(specs['max_epochs'])

    dataset = dataset.map(
        lambda image, label: SINGLE_PROCESS[dataset_name](image, label, specs, resized_size), num_parallel_calls=3)
    specs['image_size'] = resized_size

    batched_dataset = dataset.batch(specs['batch_size'])
    batched_dataset = batched_dataset.map(
        _feature_process, num_parallel_calls=3)
    batched_dataset = batched_dataset.prefetch(specs['num_gpus'])

    return batched_dataset, specs

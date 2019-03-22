import os 
import numpy as np 
import tensorflow as tf 

import dataset.dataset_save as dataset_save
import dataset.cifar10.load_cifar10_data as load_cifar10_data

def prepare_dataset(src_dir, out_dir): # TODO: label type unsure
    """
    Prepares the extraction from single dataset to .npz format.
    :param src_dir: source file directory
    :param out_dir: output file directory
    """
    try:
        train_x, train_y = load_cifar10_data.load_cifar10(src_dir, 'train')
        test_x, test_y = load_cifar10_data.load_cifar10(src_dir, 'test')
    except:
        raise ValueError("No mat files found!")
    # Save numpy arrays into .npz files
    dataset_save.save_to_npz(train_x, train_y, out_dir, 'train.npz')
    dataset_save.save_to_npz(test_x, test_y, out_dir, 'test.npz')

def _single_process(image, label, specs, resized_size):
    """
    Map function to process single instance of dataset object.
    :param image: a single instance image object, uint8, 0~255, (32, 32, 3)
    :param label: a single instance label object, uint8, 0~9, (,)
    :param specs: a dictionary that contains dataset specifications
    :param resized_size: image size after resizing
    :return feature: a dictionary contains an image and a label,
                     where image, float32, 0.~1., (resized_size, resized_size, 3),
                           label, int32, 0~9, (,)
    """
    if specs['split'] in ['train', 'test']:
        image = tf.cast(image, tf.float32) * 1. / 255.
        label = tf.cast(label, tf.int32) 

    if specs['split'] == 'train':
        image = tf.random_crop(image, [resized_size, resized_size, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    elif specs['split'] == 'test':
        image = tf.image.resize_image_with_crop_or_pad(
            image, resized_size, resized_size)
    image = tf.image.per_image_standardization(image)

    feature = {
        'image': image,
        'label': tf.one_hot(label, 10)
    }
    return feature

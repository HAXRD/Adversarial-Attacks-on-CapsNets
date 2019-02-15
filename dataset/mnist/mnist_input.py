import os 
import numpy as np 
import tensorflow as tf 
import random 

import dataset.dataset_save as dataset_save

def prepare_dataset(src_dir, out_dir): # TODO: images and labels type unsure
    """
    Prepares the extraction from single dataset to .npz format.
    :param src_dir: source file directory
    :param out_dir: output file directory
    """
    read_fn = 'mnist.npz'
    try:
        with np.load(os.path.join(src_dir, read_fn)) as f:
            train_x, train_y = f['x_train'], f['y_train']
            test_x, test_y = f['x_test'], f['y_test']
    except:
        raise ValueError("No file named '{}' found!".format(read_fn))
    # Expand images channel dimension 
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)
    # Save numpy arrays into .npz files
    dataset_save.save_to_npz(train_x, train_y, out_dir, 'train.npz')
    dataset_save.save_to_npz(test_x, test_y, out_dir, 'test.npz')

def _single_process(image, label, specs, resized_size): # TODO: label type unsure
    """
    Map function to process single instance of dataset object.
    :param image: a single instance image object, uint8, 0~255, (28, 28, 1)
    :param label: a single instance label object, uint8, 0~9, (,)
    :param specs: a dictionary that contains dataset specifications
    :param resized_size: image size after resizing
    :return feature: a dictionary contains an image and a label,
                     where image, float32, 0.~1., (resized_size, resized_size, 1),
                           label, int32, 0~9, (,)
    """
    if specs['split'] in ['train', 'test']:
        image = tf.cast(image, tf.float32) * (1. / 255.)
        label = tf.cast(label, tf.int32) 

    if specs['split'] == 'train':
        image = tf.random_crop(image, [resized_size, resized_size, 1])
        # random rotation within -15° ~ 15°
        image = tf.contrib.image.rotate(
            image, random.uniform(-0.26179938779, 0.26179938779))
    elif specs['split'] == 'test':
        image = tf.image.resize_image_with_crop_or_pad(
            image, resized_size, resized_size)

    feature = {
        'image': image, 
        'label': tf.one_hot(label, 10)
    }
    return feature

# Copyright 2018 Xu Chen All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf 
import numpy as np 
import os 
import random 

import dataset.svhn.load_svhn_data as load_svhn_data
import dataset.dataset_utils as dataset_utils

def prepare_dataset(src_dir, out_dir):
    """This function prepares extract out single dataset into npz 
    format so that the following process operations on different
    dataset can be systemized.
    Args:
        src_dir: source directory;
        out_dir: output directory.
    """
    
    """Load data from npz file"""
    try:
        train_x, train_y = load_svhn_data.load_svhn(src_dir, 'train')
        test_x, test_y = load_svhn_data.load_svhn(src_dir, 'test')
        # train_x: (73257, 32, 32, 3), train_y: (73257,)
        # test_x:  (100, 32, 32, 3), test_y:  (100,)
    except:
        raise ValueError("No mat files found!")
    
    """Save datasets into npz files"""
    dataset_utils.save_to_npz(train_x, train_y, out_dir, 'train.npz')
    dataset_utils.save_to_npz(test_x, test_y, out_dir, 'test.npz')

def _single_process(image, label, specs, resized_size):
    """Map function to process single instance of dataset object.
    
    Args:
        image: numpy array image object, (28, 28, 1), 0 ~ 255 uint8;
        label: numpy array label object, (,);
        specs: dataset specifications;
        resized_size: image size after resizing.
    Returns:
        feature: a dictionary contains image, label.
    """
    if resized_size < specs['image_size']:
        if specs['split'] == 'train':
            # random cropping 
            image = tf.random_crop(image, [resized_size, resized_size, 3])
        elif specs['split'] == 'test':
            # central cropping 
            image = tf.image.resize_image_with_crop_or_pad(
                image, resized_size, resized_size)
    # convert from 0 ~ 255 to 0. ~ 1.
    image = tf.cast(image, tf.float32) * (1. / 255.)

    feature = {
        'image': image, 
        'label': tf.one_hot(label, 10)
    }
    return feature

def _feature_process(feature):
    """Map function to process batched data inside feature dictionary.

    Args:
        feature: a dictionary contains image, label, recons_image, recons_label.
    Returns:
        batched_feature: a dictionary contains images, labels, recons_images, recons_labels.
    """
    batched_feature = {
        'images': feature['image'],
        'labels': feature['label'],
    }
    return batched_feature


def inputs(total_batch_size, num_gpus, max_epochs, resized_size, 
           data_dir, split):
    """Construct inputs for mnist dataset.

    Args:
        total_batch_size: total number of images per batch;
        num_gpus: number of GPUs available to use;
        max_epochs: maximum number of repeats;
        resized_size: image size after resizing;
        data_dir: path to the dataset;
        split: split set name after stripped out extension.
    Returns:
        batched_dataset: Dataset object, each instance is a feature dictionary;
        specs: dataset specifications.
    """
    
    """Load data from npz files"""
    assert os.path.exists(os.path.join(data_dir, '{}.npz'.format(split))) == True
    with np.load(os.path.join(data_dir, '{}.npz'.format(split))) as f:
        x, y = f['x'], f['y']
        # x: uint 8, 0 ~ 255
        # y: uint 8, 0 ~ 9
    assert x.shape[0] == y.shape[0]

    """Define specs"""
    specs = {
        'split': split, 
        'total_size': int(x.shape[0]),

        'total_batch_size': int(total_batch_size),
        'steps_per_epoch': int(x.shape[0] // total_batch_size),
        'num_gpus': int(num_gpus),
        'batch_size': int(total_batch_size / num_gpus),
        'max_epochs': int(max_epochs),

        'image_size': 32,
        'depth': 3,
        'num_classes': 10
    }

    """Process dataset object"""
    dataset = tf.data.Dataset.from_tensor_slices((x, y)) # ((32, 32, 3), (,))
    dataset = dataset.prefetch(
        buffer_size=specs['batch_size']*specs['num_gpus']*2)
    
    if split == 'train':
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
            buffer_size=specs['batch_size']*specs['num_gpus']*10,
            count=specs['max_epochs']))
    else:
        dataset = dataset.repeat(specs['max_epochs'])

    dataset = dataset.map(
        lambda image, label: _single_process(image, label, specs, resized_size), num_parallel_calls=3)
    specs['image_size'] = resized_size

    batched_dataset = dataset.batch(specs['batch_size'])
    batched_dataset = batched_dataset.map(
        _feature_process, num_parallel_calls=3)
    batched_dataset = batched_dataset.prefetch(specs['num_gpus'])

    return batched_dataset, specs

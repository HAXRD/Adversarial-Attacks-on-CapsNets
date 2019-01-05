# Copyright 2018 Xu Chen All Rights Reserved.
# Credit https://github.com/zalandoresearch/fashion-mnist#get-the-data
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

import dataset.fashion_mnist.load_fashion_mnist_data as load_fashion_mnist_data
import dataset.dataset_save as dataset_save

def prepare_dataset(src_dir, out_dir):
    """This function prepares extract out single dataset into npz 
    format so that the following process operations on different
    dataset can be systemized.
    Args:
        src_dir: source directory;
        out_dir: output directory.
    """

    """Load data from gz file"""
    try:
        train_x, train_y = load_fashion_mnist_data.load_fashion_mnist(src_dir, 'train')
        test_x, test_y = load_fashion_mnist_data.load_fashion_mnist(src_dir, 'test')
        # train_x: (?, 28, 28, 1), train_y: (?,)
        # test_x: (?, 28, 28, 1), test_y: (?,)
    except:
        raise ValueError("No gz files found!")
    

    """Process image, label dimensions"""
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)
    # train_x: (60000, 28, 28, 1), train_y: (60000,)
    # test_x:  (10000, 28, 28, 1), test_y:  (10000,)

    """Convert image range into 0. ~ 1."""
    train_x = train_x.astype(np.float32) * 1. / 255.
    test_x = test_x.astype(np.float32) * 1. / 255.

    """Save datasets into npz files"""
    dataset_save.save_to_npz(train_x, train_y, out_dir, 'train.npz')
    dataset_save.save_to_npz(test_x, test_y, out_dir, 'test.npz')

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
        # resize image
        image = tf.image.resize_images(image, [resized_size, resized_size])
    if specs['split'] == 'train':
        # random flipping
        image = tf.image.random_flip_left_right(image)
    # convert to 0. ~ 1.
    image = tf.cast(image, tf.float32)

    feature = {
        'image': image,
        'label': tf.one_hot(label, 10)
    }
    return feature
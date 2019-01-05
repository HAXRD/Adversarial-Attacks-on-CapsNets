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

import dataset.dataset_save as dataset_save

def prepare_dataset(src_dir, out_dir):
    """This function prepares extract out single dataset into npz 
    format so that the following process operations on different
    dataset can be systemized.
    Args:
        src_dir: source directory;
        out_dir: output directory.
    """
    
    """Load data from npz file"""
    read_fn = 'mnist.npz'
    try:
        with np.load(os.path.join(src_dir, read_fn)) as f:
            train_x, train_y = f['x_train'], f['y_train']
            test_x, test_y = f['x_test'], f['y_test']
            # train_x: (60000, 28, 28), train_y: (60000,)
            # test_x:  (10000, 28, 28), test_y:  (10000,)
    except:
        raise ValueError("No file named '{}' found!".format(read_fn))

    """Process image, label dimensions"""
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)
    # train_x: (60000, 28, 28, 1), train_y: (60000,)
    # test_x:  (10000, 28, 28, 1), test_y:  (10000,)

    """Convert image range into 0. ~ 1."""
    # train_x = train_x.astype(np.float32) * 1. / 255.
    # test_x = test_x.astype(np.float32) * 1. / 255.

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
    if specs['split'] == 'train':
        # random cropping
        if resized_size < specs['image_size']:
            image = tf.random_crop(image, [resized_size, resized_size])
        # random rotation within -15° ~ 15°
        image = tf.contrib.image.rotate(
            image, random.uniform(-0.26179938779, 0.26179938779))
    elif specs['split'] == 'test':
        # central cropping 
        if resized_size < specs['image_size']:
            image = tf.image.resize_image_with_crop_or_pad(
                image, resized_size, resized_size)

    if specs['split'] in ['train', 'test']:
        # convert to 0. ~ 1.
        image = tf.cast(image, tf.float32) * (1. / 255.)

    feature = {
        'image': image, 
        'label': tf.one_hot(label, 10)
    }
    return feature

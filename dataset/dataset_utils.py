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

import dataset.mnist.mnist_input as mnist_input
import dataset.fashion_mnist.fashion_mnist_input as fashion_mnist_input
import dataset.svhn.svhn_input as svhn_input
import dataset.cifar10.cifar10_input as cifar10_input

SINGLE_PROCESS = {
    'mnist': mnist_input._single_process,
    'fashion_mnist': fashion_mnist_input._single_process,
    'svhn': svhn_input._single_process,
    'cifar10': cifar10_input._single_process
}

def _feature_process(feature):
    """Map function to process batched data inside feature dictionary.

    Args:
        feature: a dictionary contains image, label.
    Returns:
        batched_feature: a dictionary contains images, labels.
    """
    batched_feature = {
        'images': feature['image'],
        'labels': feature['label']
    }
    return batched_feature


def inputs(dataset_name, total_batch_size, num_gpus, max_epochs, resized_size, 
           data_dir, split):
    """Construct inputs for mnist dataset.

    Args:
        dataset: dataset name;
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
        # x: float32, 0. ~ 1.
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

        'image_size': x.shape[1],
        'depth': x.shape[3],
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
        lambda image, label: SINGLE_PROCESS[dataset_name](image, label, specs, resized_size), num_parallel_calls=3)
    specs['image_size'] = resized_size

    batched_dataset = dataset.batch(specs['batch_size'])
    batched_dataset = batched_dataset.map(
        _feature_process, num_parallel_calls=3)
    batched_dataset = batched_dataset.prefetch(specs['num_gpus'])

    return batched_dataset, specs

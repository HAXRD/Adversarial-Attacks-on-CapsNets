from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import shutil

import dataset.mnist.mnist_input as mnist_input
import dataset.cifar10.cifar10_input as cifar10_input

PREPARE_DATASET = {
    'mnist': mnist_input.prepare_dataset,
    'cifar10': cifar10_input.prepare_dataset
}

def init_unified_datasets(src_dir, out_dir):
    """
    Prepares datasets for three different model settings.
    :param src_dir: source file directory
    :param out_dir: output file directory
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    model_types = ['cnn', 'caps', 'caps_r']
    datasets = ['mnist', 'cifar10']
    for model_type in model_types:
        for dataset in datasets:
            created_dir = os.path.join(out_dir, model_type, dataset)
            os.makedirs(created_dir)
            PREPARE_DATASET[dataset](
                os.path.join(src_dir, dataset), 
                created_dir)

if __name__ == "__main__":
    init_unified_datasets('dataset', '../scratch/data')
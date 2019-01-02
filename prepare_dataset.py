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

from __future__ import absolute_import, division, print_function

import os

import shutil

from dataset.mnist import mnist_input
from dataset.svhn import svhn_input

PREPARE_DATASET = {
    'mnist': mnist_input.prepare_dataset,
    'svhn': svhn_input.prepare_dataset
}

def init_unified_datasets(src_dir, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    model_types = ['cnn', 'caps', 'caps_r']
    # datasets = ['mnist', 'fashion_mnist', 'svhn', 'cifar10']
    datasets = ['mnist', 'svhn']
    for model_type in model_types:
        for dataset in datasets:
            created_dir = os.path.join(out_dir, model_type, dataset)
            os.makedirs(created_dir)
            PREPARE_DATASET[dataset](
                os.path.join(src_dir, dataset), 
                created_dir)

if __name__ == "__main__":
    init_unified_datasets('dataset', 'data')
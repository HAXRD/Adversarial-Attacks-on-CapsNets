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
"""Tests for experiment"""

import os 
import shutil
import numpy as np 
import tensorflow as tf 

from experiment import test

class ExpTestTest(tf.test.TestCase):

    """Test without adversarial examples"""
    def testTestMNIST(self):
        test(num_gpus=2, 
             total_batch_size=200, image_size=28,
             summary_dir='debug/summary/caps/mnist/Default/',
             load_test_path='debug/data/caps/mnist/test.npz')

    def testTestSVHN(self):
        test(num_gpus=2,
             total_batch_size=200, image_size=28,
             summary_dir='debug/summary/caps/svhn/Default/',
             load_test_path='debug/data/caps/svhn/test.npz')

if __name__ == '__main__':
    tf.test.main()
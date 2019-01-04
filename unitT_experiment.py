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
import numpy as np 
import tensorflow as tf 

from experiment import *

hparams = default_hparams()

class ExperimentTest(tf.test.TestCase):

    """Train without adversarial training"""
    def testTrainMNIST(self):
        train(hparams, num_gpus=2, data_dir='debug/data/caps/mnist', dataset='mnist', 
                       adversarial_method='Default',
                       model_type='caps', total_batch_size=200, image_size=28,
                       summary_dir='debug/summary/', save_epochs=2, max_epochs=4)

    def testTrainSVHN(self):
        train(hparams, num_gpus=2, data_dir='debug/data/caps/svhn', dataset='svhn', 
                       adversarial_method='Default',
                       model_type='caps', total_batch_size=200, image_size=28,
                       summary_dir='debug/summary/', save_epochs=2, max_epochs=4)
        

    """Test without adversarial examples"""
    def testTestMNIST(self):
        pass

    def testTestSVHN(self):
        pass

if __name__ == '__main__':
    tf.test.main()
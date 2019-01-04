
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

from experiment import gen_adv

class ExpGenAdvTest(tf.test.TestCase):

    """Generate adversarial exmaples"""
    def testGenAdvMNIST(self):
        gen_adv(num_gpus=2, data_dir='debug/data/caps/mnist/', dataset='mnist',
                adversarial_method='FGSM',
                total_batch_size=2, image_size=28,
                summary_dir='debug/summary/caps/mnist/Default/', all_=2)

        with np.load(os.path.join('debug/data/caps/mnist/', 'test_FGSM.npz')) as f:
            images, labels = f['x'], f['y']
        
        self.assertEqual((4, 28, 28, 1), images.shape)
        self.assertEqual((4,), labels.shape)

    def testGenAdvSVHN(self):
        gen_adv(num_gpus=2, data_dir='debug/data/caps/svhn',
        dataset='svhn',
                adversarial_method='FGSM',
                total_batch_size=2, image_size=28, 
                summary_dir='debug/summary/caps/svhn/Default/', all_=2)

        with np.load(os.path.join('debug/data/caps/svhn/', 'test_FGSM.npz')) as f:
            images, labels = f['x'], f['y']
        
        self.assertEqual((4, 28, 28, 3), images.shape)
        self.assertEqual((4,), labels.shape)

if __name__ == "__main__":
    tf.test.main()
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
"""Tests for mnist_input"""

import numpy as np 
import tensorflow as tf 

import dataset.mnist.mnist_input as mnist_input

MNIST_DATA_DIR = './dataset/mnist'

class MnistInputTest(tf.test.TestCase):
    
    def testPrepareDataset(self):
        mnist_input.prepare_dataset(MNIST_DATA_DIR)

    def testTrain(self):
        with self.test_session(graph=tf.Graph()) as sess:
            batched_dataset, specs = mnist_input.inputs(
                total_batch_size=200,
                num_gpus=2,
                max_epochs=1,
                resized_size=28,
                data_dir=MNIST_DATA_DIR,
                split='train')
            iterator = batched_dataset.make_initializable_iterator()
            sess.run(iterator.initializer)
            batch_data = iterator.get_next()
            batch_val = sess.run(batch_data)
            images, labels = batch_val['images'], batch_val['labels']

            self.assertEqual((100, 10), labels.shape)
            self.assertItemsEqual([0, 1], np.unique(labels))
            self.assertEqual(28, specs['image_size'])
            self.assertEqual((100, 28, 28, 1), images.shape)

if __name__ == '__main__':
    tf.test.main()
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

import os
import numpy as np 
import tensorflow as tf 

import dataset.svhn.svhn_input as svhn_input

src_dir = './dataset'
out_dir = './debug/data'
class MnistInputTest(tf.test.TestCase):
    
    def testPrepareDataset(self):
        model_types = ['cnn', 'caps', 'caps_r']
        dataset = 'svhn'
        for model_type in model_types:
            created_dir = os.path.join(out_dir, model_type, dataset)
            svhn_input.prepare_dataset(
                os.path.join(src_dir, dataset),
                created_dir)

    def testTrain(self):
        with self.test_session(graph=tf.Graph()) as sess:
            batched_dataset, specs = svhn_input.inputs(
                total_batch_size=200,
                num_gpus=2,
                max_epochs=1,
                resized_size=28,
                data_dir=os.path.join(out_dir, 'caps', 'svhn'),
                split='train')
            iterator = batched_dataset.make_initializable_iterator()
            sess.run(iterator.initializer)
            batch_data = iterator.get_next()
            batch_val = sess.run(batch_data)
            images, labels = batch_val['images'], batch_val['labels']

            self.assertEqual((100, 10), labels.shape)
            self.assertItemsEqual([0, 1], np.unique(labels))
            self.assertEqual(28, specs['image_size'])
            self.assertEqual((100, 28, 28, 3), images.shape)

if __name__ == '__main__':
    tf.test.main()
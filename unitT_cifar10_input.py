"""Tests for cifar10_input"""

import os
import numpy as np 
import tensorflow as tf 

import dataset.cifar10.cifar10_input as cifar10_input
import dataset.dataset_utils as dataset_utils

src_dir = './dataset'
out_dir = './debug/data'
class Cifar10InputTest(tf.test.TestCase):
    
    def testPrepareDataset(self):
        model_types = ['cnn', 'caps', 'caps_r']
        dataset = 'cifar10'
        for model_type in model_types:
            created_dir = os.path.join(out_dir, model_type, dataset)
            cifar10_input.prepare_dataset(
                os.path.join(src_dir, dataset),
                created_dir)

    def testTrain(self):
        with self.test_session(graph=tf.Graph()) as sess:
            batched_dataset, specs = dataset_utils.inputs(
                dataset_name='cifar10',
                total_batch_size=200,
                num_gpus=2,
                max_epochs=1,
                resized_size=28,
                data_dir=os.path.join(out_dir, 'caps', 'cifar10'),
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
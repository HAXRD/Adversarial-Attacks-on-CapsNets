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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import model
from models.layers import variables

class CNNModel(model.Model):
    """
    A baseline multi GPU Model without capsule layers.
    The inference graph includes ReLU convolution layers and fully connected
    layers. The last layer is linear and has 10 units.
    """

    def _add_convs(self, input_tensor, channels, tower_idx):
        """
        Adds the convolution layers.
        Adds a series of convolution layers with ReLU nonlinearity and pooling
        after each of them.
        :param input_tensor: a 4D tensor as the input to the first Conv layer
        :param channels: a list of channel sizes for input_tensor and following
                         conv layers. Number of channels in input tensor should 
                         be equal to channels[0]
        :param tower_idx: the index number for this tower. Each tower is named
                          as tower_{tower_idx} and resides on 'gpu:{tower_idx}'
        :return: a 4D tensor as the output of the last pooling layer
        """
        for i in range(1, len(channels)):
            with tf.variable_scope('conv{}'.format(i)) as scope:
                kernel = variables.weight_variable(
                    shape=[5, 5, channels[i - 1], channels[i]], stddev=5e-2,
                    verbose=self._hparams.verbose)
                conv = tf.nn.conv2d(
                    input_tensor,
                    kernel, [1, 1, 1, 1],
                    padding=self._hparams.padding,
                    data_format='NCHW')
                biases = variables.bias_variable([channels[i]],
                                                 verbose=self._hparams.verbose)
                pre_activation = tf.nn.bias_add(
                    conv, biases, data_format='NCHW', name='logits')
                
                relu = tf.nn.relu(pre_activation, name=scope.name)
                if self._hparams.verbose:
                    tf.summary.histogram('activation', relu)
                input_tensor = tf.contrib.layers.max_pool2d(
                    relu, kernel_size=2, stride=2, data_format='NCHW', padding='SAME')
        
        return input_tensor

    def build_replica(self, tower_idx):
        """
        Adds a replica graph ops.
        Builds the architecture of the neural net to derive logits from 
        batched_dataset. The inference graph defined here should involve 
        trainable variables otherwise the optimizer will raise a ValueError.
        :param tower_idx: the index number for this tower. Each tower is named
                          as tower_{tower_idx} and resides on 'gpu:{tower_idx}'
        :return: an Inferred namedtuple contains (logits, None)
        """
        # Image specs
        image_size = self._specs['image_size']
        image_depth = self._specs['depth']
        num_classes = self._specs['num_classes']

        # Define input_tensor for input batched_images
        batched_images = tf.placeholder(tf.float32, 
            shape=[None, image_size, image_size, image_depth], 
            name='batched_images')
        tf.add_to_collection(
            'tower_%d_batched_images' % tower_idx, batched_images) #! visual

        batched_images_splits = tf.split(
            batched_images, num_or_size_splits=self._specs['batch_size'], axis=0) 
        for i in range(self._specs['batch_size']):
            tf.add_to_collection(
                'tower_%d_batched_images_split' % tower_idx, batched_images_splits[i])
        
        batched_images = tf.concat(batched_images_splits, axis=0)
        batched_images = tf.transpose(batched_images, [0, 3, 1, 2])
        
        # Add convolutional layers
        conv_out = self._add_convs(batched_images, [image_depth, 512, 256], tower_idx)
        hidden1 = tf.contrib.layers.flatten(conv_out) # flatten neurons, shape (?, rest)

        # Add fully connected layer 1, activation = relu
        with tf.variable_scope('fc1') as scope:
            dim = hidden1.get_shape()[1].value
            weights = variables.weight_variable(shape=[dim, 1024], stddev=0.1,
                                                verbose=self._hparams.verbose)
            biases = variables.bias_variable(shape=[1024],
                                             verbose=self._hparams.verbose)
            pre_activation = tf.add(tf.matmul(hidden1, weights), biases, name='logits')

            hidden2 = tf.nn.relu(pre_activation, name=scope.name)
        
        # Add fully connected layer 2, activation = None
        with tf.variable_scope('softmax_layer') as scope:
            weights = variables.weight_variable(
                shape=[1024, num_classes], stddev=0.1,
                verbose=self._hparams.verbose)
            biases = variables.bias_variable(
                shape=[num_classes],
                verbose=self._hparams.verbose)
            logits = tf.add(tf.matmul(hidden2, weights), biases, name='logits')
        
        # Declare one-hot format placeholder for batched_labels
        batched_labels = tf.placeholder(tf.int32,
            shape=[None, num_classes], name='batched_labels') 
        tf.add_to_collection('tower_%d_batched_labels' % tower_idx, batched_labels)

        return model.Inferred(logits, None)




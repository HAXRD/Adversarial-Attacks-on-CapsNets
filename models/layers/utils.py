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

def _margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
    """
    Penalizes deviations from margin for each logit.
    Each wrong logit costs its distance to margin. For negative logits margin
    is 0.1 and for positives it is 0.9. First subtract 0.5 from all logits. 
    Now margin is 0.4 from each side.
    :param labels: one hot encoding of the ground truth
    :param raw_logits: model predictions in range [0, 1]
    :param margin: the margin after substracting 0.5 from raw_logits
    :param down_weight: the factor for negative cost
    :return: total loss for each instance
    """
    labels = tf.cast(labels, tf.float32)
    logits = raw_logits - 0.5
    positive_cost = labels * tf.cast(tf.less(logits, margin),
                                     tf.float32) * tf.pow(logits - margin, 2)
    negative_cost = (1 - labels) * tf.cast(tf.greater(logits, -margin),
                                     tf.float32) * tf.pow(logits + margin, 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost

def evaluate(tower_idx, logits, scope, loss_type):
    """
    Calculates total loss and performance metrics.
    :param tower_idx: the index of the tower. Each tower is named as 
                      tower_{tower_idx} and resides on gpu:{tower_idx}
    :param logits: model output
    :param scope: the scope name to collect losses of
    :param loss_type: 'softmax' or 'margin'
    :return: total_loss of model and number of correct predictions per batch
    """
    labels = tf.get_default_graph().get_tensor_by_name(scope + 'batched_labels:0') # 'tower_{i}/batched_labels:0'
    with tf.name_scope('loss'): # 'tower_i/loss'
        if loss_type == 'softmax':
            classification_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        elif loss_type == 'margin':
            classification_loss = _margin_loss(labels=labels, raw_logits=logits)
        else:
            raise ValueError('No loss type found as {}'.format(loss_type))
        individual_classification_loss = tf.reduce_mean(classification_loss, axis=1)
        tf.add_to_collection('tower_%d_BIM_loss' % tower_idx, individual_classification_loss)

        # compute least-likely class loss
        least_likely_class_loss = tf.reduce_min(logits, axis=1)
        tf.add_to_collection('tower_%d_ILLCM_loss' % tower_idx, least_likely_class_loss)

        with tf.name_scope('total'): # 'tower_i/loss/total'
            batch_classification_loss = tf.reduce_mean(classification_loss)
            tf.add_to_collection('losses', batch_classification_loss)
    
    tf.summary.scalar('batch_classification_loss', batch_classification_loss)

    all_losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(all_losses, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)

    with tf.name_scope('accuracy'): # 'tower_i/accuracy'
        with tf.name_scope('correct_predictions'): # 'tower_i/accuracy/correct_predictions'
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            lbls = tf.argmax(labels, axis=1, output_type=tf.int32)
            correct = tf.cast(tf.equal(preds, lbls), tf.float32)
            num_correct_per_batch = tf.reduce_sum(correct)
            accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('correct_predictions_per_batch', num_correct_per_batch)
    tf.summary.scalar('accuracy', accuracy)
    return total_loss, num_correct_per_batch, accuracy
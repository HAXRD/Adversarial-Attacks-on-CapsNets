
# Credit: https://github.com/gongzhitaao/tensorflow-adversarial
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

import tensorflow as tf 

def compute_one_step_adv(loss, xs_split, batch_size, eps=0.01, clip_min=0., clip_max=1.):
    """This function receives a tensor: {loss} and a list of tensor: {xs_split}
    as input. Then we split {loss} into {batch_size} splits and compute one-step
    adversarial procedure.

    Args:
        loss: corresponding loss function, cross_entropy, margin, least-likely class
            shape (?,)
        xs_split: input batched images tensor, 
            shape [(1, 28, 28, 1), ...]
        batche_size: ?
    """
    
    loss_split = tf.split(loss, num_or_size_splits=batch_size, axis=0)

    xs_advs = []
    for k in range(batch_size):
        dy_dx = tf.gradients(loss_split[k], xs_split[k])[0]
        xs_adv = tf.stop_gradient(xs_split[k] + 1/255. * eps * tf.sign(dy_dx))
        xs_adv = tf.clip_by_value(xs_adv, clip_min, clip_max)
        xs_advs.append(xs_adv)
        print("Done {}/{}...".format(k+1, batch_size), end='\r')
    xs_adv_out = tf.concat(xs_advs, axis=0)

    return xs_adv_out


def compute_one_step_adv_avg(loss, xs, batch_size, eps=0.01, clip_min=0., clip_max=1.):
    
    avg_loss = tf.reduce_mean(loss, axis=0)

    dy_dxs = tf.gradients(avg_loss, xs)[0] 
    xs_adv = tf.stop_gradient(xs + 1./255. * eps * tf.sign(dy_dxs))
    xs_adv = tf.clip_by_value(xs, clip_min, clip_max)

    return xs_adv

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

def FGSM(loss, x, eps=0.01, clip_min=0., clip_max=1.):
    
    xadv = tf.identity(x)

    xgrad = tf.gradients(loss, x)

    xadv = xadv + eps*tf.sign(xgrad)

    xadv = tf.clip_by_value(xadv, clip_min, clip_max)

    return xadv

# def BIM(loss, xs, batch_size, eps=0.01, epochs=1, clip_min=0., clip_max=1.):
#     """This function receives two tensors, input batched images tensor: {xs}, 
#     and selected loss function tensor: {loss}. It first make a tensor copy of 
#     the input image tensor, split {xs} and {loss} in axis=0 into {batch_size}
#     splits, then compute their gradients and apply adversarial procedure iteratively.

#     Args:
#         loss: corresponding loss function, cross_entropy, margin, least-likely class
#             shape (?,)
#         xs: input batched images tensor, 
#             shape (?, 28, 28, 1)
#         batche_size: ?
#     """

#     xs_adv = tf.identity(xs)
#     xs_adv_split = tf.split(xs_adv, num_or_size_splits=batch_size, axis=0)

#     loss_split = tf.split(loss, num_or_size_splits=batch_size, axis=0)

#     def _cond(xs_adv_split, i):
#         return tf.less(i, epochs)

#     def _body(xs_adv_split, i):
#         for k in range(batch_size):
#             dy_dx, = tf.gradients(loss_split[k], xs_adv_split[k])
#             xs_adv_split[k] = tf.stop_gradient(xs_adv_split[k] + eps*tf.sign(dy_dx))
#             xs_adv_split[k] = tf.clip_by_value(xs_adv_split[k], clip_min, clip_max)
#         return xs_adv_split, i+1

#     xs_adv_split, _ = tf.while_loop(_cond, _body, (xs_adv_split, 0), back_prop=False, name='adversarial_procedure')

#     xs_adv_out = tf.concat(xs_adv_split, 0)

#     return xs_adv_out

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
    xs_split_cp = tf.identity(xs_split)
    loss_split = tf.split(loss, num_or_size_splits=batch_size, axis=0)

    for k in range(batch_size):
        dy_dx = tf.gradients(loss_split[k], xs_split[k])[0]
        xs_split_cp[k] = tf.stop_gradient(xs_split_cp[k] + eps * tf.sign(dy_dx))
        xs_split_cp[k] = tf.clip_by_value(xs_split_cp[k], clip_min, clip_max)

    xs_adv_out = tf.concat(xs_split_cp, axis=0)

    return xs_adv_out
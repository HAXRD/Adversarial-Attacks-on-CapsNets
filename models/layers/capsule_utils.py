# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Library for capsule layers."""

from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np 
import tensorflow as tf 

from models.layers import variables

def _squash(in_tensor):
    """
    Applies (squash) to capsule layer.
    :param in_tensor: tensor,
                      [batch, num_cap_types, num_atoms] for a fc capsule layer or 
                      [batch, num_cap_types, num_atoms, h, w] for a convolutional 
                      capsule layer
    :return: a tensor with same shape but squashed
    """
    with tf.name_scope('norm_non_linearity'):
        norm = tf.norm(in_tensor, axis=2, keepdims=True)
        norm_squared = norm * norm
        return (in_tensor / norm) * (norm_squared / (1 + norm_squared))

def _leaky_routing(logits, out_dim):
    """
    Adds extra dimmension to routing logits.
    This enables active capsules to be routed to the extra dim if they are not a
    good fit for any of the capsules in the layer above.
    :param logits: the original logits, shape [in_dim, out_dim] if fully connected,
                   otherwise, it has two more dimmensions
    :param out_dim: the dimension of output capsule
    :return: routing probabilities for each pair of capsules, which has the same
             shape as logits
    """
    leak = tf.zeros_like(logits, optimize=True)
    leak = tf.reduce_sum(leak, axis=2, keepdims=True)
    leaky_logits = tf.concat([leak, logits], axis=2)
    leaky_routing = tf.nn.softmax(leaky_logits, axis=2)
    return tf.split(leaky_routing, [1, out_dim], 2)[1]

def _update_routing(tower_idx, votes, biases, logit_shape, 
                    num_ranks, in_dim, out_dim, reassemble, 
                    leaky, num_routing):
    """
    Sums over scaled votes and applies squash to compute the activations.
    Iteratively updates routing logits (scales) based on the similarity between
    the activation of this layer and the votes of the layer below.
    :param tower_idx: the index number for this tower. Each tower is named as 
                      tower_{tower_idx} and resides on gpu:{tower_idx}
    :param votes: the transformed outputs of the layer below
    :param biases: bias variable
    :param logit_shape: shape of the logit to be initialized
    :param num_ranks: rank of the votes tensor. For fully connected capsule it
                      is 4, for convolutional capsule it is 6
    :param in_dim: number of capsule types of input
    :param out_dim: number of capsule types of output
    :param leaky: whether to use leaky routing
    :param num_routing: number of routing iterations
    :param reassemble: whether to use reassemble method
    :return: the activation tensor of the output layer after {num_routing} iterations
    """
    votes_t_shape = [3, 0, 1, 2]
    r_t_shape = [1, 2, 3, 0]
    for i in range(num_ranks - 4):
        votes_t_shape += [i + 4]
        r_t_shape += [i + 4]
    votes_trans = tf.transpose(votes, votes_t_shape)

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, in_dim, out_dim, ...]
        if leaky:
            route = _leaky_routing(logits, out_dim)
        else:
            route = tf.nn.softmax(logits, axis=2)
        preact_unrolled = route * votes_trans
        preact_trans = tf.transpose(preact_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = _squash(preactivate)
        activations = activations.write(i, activation)
        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_ranks, dtype=np.int32).tolist()
        tile_shape[1] = in_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        # logits = logits.write(i+1, logit + distances)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)

    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing - 1,
        _body, 
        loop_vars=[i, logits, activations],
        swap_memory=True)

    # do it manually
    if leaky:
        route = _leaky_routing(logits, out_dim)
    else:
        route = tf.nn.softmax(logits, axis=2) # (?, 512, 10)
    """Normal route section"""
    preact_unrolled = route * votes_trans
    preact_trans = tf.transpose(preact_unrolled, r_t_shape)
    preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
    activation = _squash(preactivate)
    activations = activations.write(num_routing - 1, activation)
    act_3d = tf.expand_dims(activation, 1)
    tile_shape = np.ones(num_ranks, dtype=np.int32).tolist()
    tile_shape[1] = in_dim
    act_replicated = tf.tile(act_3d, tile_shape)
    distances = tf.reduce_sum(votes * act_replicated, axis=3)
    logits += distances

    full_norm = tf.norm(preactivate, axis=2, keepdims=True)
    full_norm_squared = full_norm * full_norm
    scale = full_norm / (1 + full_norm_squared)
    if reassemble:
        """Boost section"""
        # transpose route to make compare easier
        route_trans = tf.transpose(route, [0, 2, 1])
        ten_splits = tf.split(route_trans, num_or_size_splits=10, axis=1)
        threshold = tf.get_collection('tower_%d_batched_threshold' % tower_idx)[0]
        for split in ten_splits:
            split_shape = tf.shape(split) # (?, 1, 512)
            valid_cap_indices = tf.less_equal(
                split, 
                tf.fill(split_shape, threshold)) # threshold here
            valid_cap_indices_sq = tf.squeeze(valid_cap_indices)
            valid_cap_multiplier = tf.cast(valid_cap_indices_sq, tf.float32) # (?, 512) 1.0 or 0.0
            valid_cap_multiplier_tiled = tf.tile(
                tf.expand_dims(valid_cap_multiplier, -1), 
                [1, 1, 10])
            preact_unrolled = valid_cap_multiplier_tiled * route * votes_trans
            preact_trans = tf.transpose(preact_unrolled, r_t_shape)
            preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
            # activation = _squash(preactivate)
            # manual squash
            with tf.name_scope('manual_norm_non_linearity'):
                activation = preactivate * scale
            act_norm = tf.norm(activation, axis=-1, name='act_norm')
            tf.add_to_collection('tower_%d_ensemble_acts' % tower_idx, act_norm) # total 10

    return activations.read(num_routing - 1)
    

def _depthwise_conv3d(tower_idx, in_tensor, in_dim, in_atoms,
                      out_dim, out_atoms,
                      kernel, stride=2, padding='SAME'):
    """
    Perform 2D convolution given a 5D input tensor.
    This layer given an input tensor of shape (batch, in_dim, in_atoms, in_h, in_w).
    We squeeze this first two dimmensions to get a 4R tensor as the input of 
    tf.nn.conv2d. Then splits the first dimmension and the last dimmension and 
    returns the 6R convolution output.
    :param tower_idx: the index number for this tower. Each tower is named as 
                      tower_{tower_idx} and resides on gpu:{tower_idx}
    :param in_tensor: a 5D tensor, whose last two dimensions representing height and width
    :param in_dim: number of capsule types of input
    :param in_atoms: number of units of each input capsule
    :param out_dim: number of capsule types of output
    :param out_atoms: number of units of each output capsule 
    :param kernel: convolutional kernel variable
    :param stride: stride of the convolutional kernel
    :param padding: 'SAME' or 'VALID', padding mechanism for convolutional kernels
    :return: 6D tensor output of a 2D convolution with the shape of [batch, in_dim, out_dim,
             out_atoms, out_h, out_w], the convolution output shape and input shape
    """
    with tf.name_scope('conv'):
        in_shape = tf.shape(in_tensor) # op
        _, _, _, in_height, in_width = in_tensor.get_shape() # (batch, in_dim, in_atoms, in_h, in_w)
        # Reshape in_tensor to 4R by merging first two dimmensions.
        in_tensor_reshaped = tf.reshape(in_tensor, [
            in_shape[0]*in_dim, in_atoms, in_shape[3], in_shape[4]
        ])
        in_tensor_reshaped.set_shape((None, in_atoms, in_height.value, in_width.value))
        
        # do convolution
        conv = tf.nn.conv2d(
            in_tensor_reshaped,
            kernel, [1, 1, stride, stride],
            padding=padding,
            data_format='NCHW')
        conv_shape = tf.shape(conv) # shape (batch*in_dim, out_dim*out_atoms, H, W)
        _, _, conv_height, conv_width = conv.get_shape() 
        # Reshape back to 6R by splitting first dimmension to batch and in_dim
        # and splitting the second dimmension to out_dim and out_atoms.
        
        conv_reshaped = tf.reshape(conv, [
            in_shape[0], in_dim, out_dim, out_atoms, conv_shape[2], conv_shape[3]
        ], name='votes')
        conv_reshaped.set_shape((None, in_dim, out_dim, out_atoms, 
            conv_height.value, conv_width.value))

        return conv_reshaped, conv_shape, in_shape
        
def conv_slim_capsule(tower_idx, in_tensor, in_dim, in_atoms,
                      out_dim, out_atoms, layer_name,
                      kernel_size=5, stride=2, padding='SAME', 
                      reassemble=False,
                      **routing_args):
    """
    Builds a slim convolutional capsule layer.
    This layer performs 2D convolution given 5R input tensor of shape
    (batch, in_dim, in_atoms, in_h, in_w). Then refines the votes with 
    routing and applies Squash nonlinearity for each capsule.
    Each capsule in this layer is a convolutional unit and shares its kernel
    over its positional grid (e.g. 9x9) and different capsules below. Therefore,
    number of trainable variables in this layer is:

        kernel: (kernel_size, kernel_size, in_atoms, out_dim * out_atoms)
        bias: (out_dim, out_atoms)
    
    Output of a conv2d layer is a single capsule with channel number of atoms.
    Therefore conv_slim_capsule is suitable to be added on top of a conv2d layer
    with num_routing=1, in_dim=1 and in_atoms = conv_channels.
    :param tower_idx: the index number for this tower. Each tower is named as 
                      tower_{tower_idx} and resides on gpu:{tower_idx}
    :param in_tensor: 5D tensor, last two dimmensions representing height and width
    :param in_dim: number of capsule types of input
    :param in_atoms: number of units of each input capsule
    :param out_dim: number of capsule types of output
    :param out_atoms: number of units of each output capsule 
    :param layer_name: name of this layer
    :param kernel_size: convolutional kernel size [kernel_size, kernel_size]
    :param stride: stride of the convolutional kernel
    :param padding: 'SAME' or 'VALID', padding mechanism for convolutional kernels
    :param **routing_args: dictionary {leaky, num_routing}, args to be passed to 
                           the routing procedure
    :return: tensor of activations for this layer of shape [batch, out_dim, out_atoms,
             out_h, out_w]
    """
    with tf.variable_scope(layer_name):
        kernel = variables.weight_variable(
            shape=[kernel_size, kernel_size, in_atoms, out_dim * out_atoms])
        biases = variables.bias_variable(
            shape=[out_dim, out_atoms, 1, 1]) 
        votes, votes_shape, in_shape = _depthwise_conv3d(
            tower_idx, in_tensor, in_dim, in_atoms, out_dim, out_atoms, kernel, stride, padding)
        
        with tf.name_scope('routing'):
            logit_shape = tf.stack([
                in_shape[0], in_dim, out_dim, votes_shape[2], votes_shape[3]
            ])
            biases_replicated = tf.tile(biases, [1, 1, votes_shape[2], votes_shape[3]])

            activations = _update_routing(
                tower_idx,
                votes=votes, 
                biases=biases_replicated, 
                logit_shape=logit_shape, 
                num_ranks=6, 
                in_dim=in_dim, 
                out_dim=out_dim,
                reassemble=reassemble,
                **routing_args)
        return activations

def capsule(tower_idx, in_tensor, in_dim, in_atoms,
            out_dim, out_atoms, layer_name,
            reassemble,
            **routing_args):
    """
    Builds a fully connected capsule layer.
    Given an input tensor of shape (batch, in_dim, in_atoms), this op
    performs the following:

        1. For each input capsule, multiplies it with the weight variables
        to get votes of shape (batch, in_dim, out_dim, out_atoms);
        2. Scales the votes for each output capsule by routing;
        3. Squashes the output of each capsule to have norm less than one.

    Each capsule of this layer has one weight tensor for each capsule of
    layer below. Therefore, this layer has the following number of 
    trainable variables:
        kernel: (in_dim, in_atoms, out_dim * out_atoms)
        biases: (out_dim, out_atoms)
    
    :param in_tensor: 5D tensor, last two dimmensions representing height and width
    :param in_dim: number of capsule types of input
    :param in_atoms: number of units of each input capsule
    :param out_dim: number of capsule types of output
    :param out_atoms: number of units of each output capsule 
    :param layer_name: name of this layer
    :param **routing_args: dictionary {leaky, num_routing}, args for routing
    :return tensor of activations for this layer of shape [batch, out_dim, out_atoms]
    """
    with tf.variable_scope(layer_name):
        weights = variables.weight_variable(
            [in_dim, in_atoms, out_dim * out_atoms])
        biases = variables.bias_variable([out_dim, out_atoms])
        with tf.name_scope('Wx_plus_b'):
            # Depthwise matmul: [b, d, c] @ [d, c, o_c] = [b, d, o_c]
            # to do this: tile input, do element-wise multiplication and reduce
            # sum over in_atoms dimmension.
            in_tiled = tf.tile(
                tf.expand_dims(in_tensor, -1), 
                [1, 1, 1, out_dim * out_atoms])
            votes = tf.reduce_sum(in_tiled * weights, axis=2)
            votes_reshaped = tf.reshape(votes,
                                        [-1, in_dim, out_dim, out_atoms])
        
        with tf.name_scope('routing'):
            in_shape = tf.shape(in_tensor)
            logit_shape = tf.stack([in_shape[0], in_dim, out_dim])
            activations = _update_routing(
                tower_idx,
                votes=votes_reshaped,
                biases=biases, 
                logit_shape=logit_shape, 
                num_ranks=4, 
                in_dim=in_dim, 
                out_dim=out_dim,
                reassemble=reassemble,
                **routing_args)
        
        return activations
    
def reconstruction(capsule_mask, num_atoms, capsule_embedding, layer_sizes,
                   num_pixels, reuse, image, balance_factor):
    """
    Adds the reconstruction loss and calculates the reconstructed image.
    Given the last capsule output layer as input of shape (batch, 10, num_atoms),
    add 3 fully connected layers on top of it.
    Feeds the masked output of the model to the reconstruction sub-network.
    Adds the difference with reconstruction image as reconstruction loss to the 
    loss collection.
    :param capsule_mask: for each intance in the batch it has the one-hot 
                         encoding of the target id
    :param num_atoms: number of atoms in the given capsule_embedding
    :param capsule_embedding: output of the last capsule layer
    :param layer_sizes: (scalar, scalar), size of the first and second layer
    :param num_pixels: number of pixels in the target image
    :param reuse: whether to set reuse variable
    :param image: the reconstruction target image
    :param balance_factor: downweight the loss to be in the valid range
    :return: the reconstruction images of shape [batch_size, num_pixels]
    """
    first_layer_size, second_layer_size = layer_sizes
    capsule_mask_3d = tf.expand_dims(tf.cast(capsule_mask, tf.float32), -1)
    atom_mask = tf.tile(capsule_mask_3d, [1, 1, num_atoms])

    filtered_embedding = capsule_embedding * atom_mask
    filtered_embedding_2d = tf.contrib.layers.flatten(filtered_embedding)
    
    reconstruction_2d = tf.contrib.layers.stack(
        inputs=filtered_embedding_2d,
        layer=tf.contrib.layers.fully_connected,
        stack_args=[(first_layer_size, tf.nn.relu),
                    (second_layer_size, tf.nn.relu), 
                    (num_pixels, tf.sigmoid)],
        reuse=reuse,
        scope='recons',
        weights_initializer=tf.truncated_normal_initializer(
            stddev=0.1, dtype=tf.float32),
        biases_initializer=tf.constant_initializer(0.1))
    
    with tf.name_scope('loss'):
        image_2d = tf.contrib.layers.flatten(image)
        distance = tf.pow(reconstruction_2d - image_2d, 2)
        loss = tf.reduce_sum(distance, axis=-1)
        batch_loss = tf.reduce_mean(loss)
        balanced_loss = balance_factor * batch_loss
        tf.add_to_collection('losses', balanced_loss)
        tf.summary.scalar('reconstruction_error', balanced_loss)

    return reconstruction_2d

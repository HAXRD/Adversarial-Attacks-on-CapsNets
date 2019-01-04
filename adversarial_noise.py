
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

def BIM(loss, x, eps=0.01, epochs=1, clip_min=0., clip_max=1.):
    pass

def LLCM():
    pass
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

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_gpus', 2,
                        'Number of GPUs available.')
tf.flags.DEFINE_integer('total_batch_size', 200,
                        'The total batch size for each batch. It will be splitted into num_gpus partitions.')
tf.flags.DEFINE_integer('save_epochs', 5,
                        'How often to save ckpt files.')
tf.flags.DEFINE_integer('max_epochs', 500,
                        'train, evaluate, ensemble: maximum epochs to run;\n'
                        'others: number of different examples to viasualize.')
tf.flags.DEFINE_integer('image_size', 28,
                        'Define the image size for dataset.')

tf.flags.DEFINE_string('mode', 'train',
                       'train: train the model;\n'
                       'test: test the model;\n'
                       'evaluate: evaluate the model for both training and testing set using different evaluation metrics;\n')
tf.flags.DEFINE_string('adversarial_method', 'Default',
                       'Default: no adversarial training method applied;\n'
                       'FGSM: fast gradient sign method;\n'
                       'BIM: basic iterative method;\n'
                       'LLCM: least-likely class method.')
tf.flags.DEFINE_string('hparams_override', None,
                       '--hparams_override=num_prime_capsules=64,padding=SAME,leaky=true,remake=false')
tf.flags.DEFINE_string('data_dir', 'dataset',
                       'The data directory')
tf.flags.DEFINE_string('dataset', 'mnist',
                       'The dataset to use for the experiment.\n'
                       'mnist, fashion_mnist, svhn, cifar10.')
tf.flags.DEFINE_string('model', 'cap',
                       'The model to use for the experiment.\n'
                       'cap or cnn.')
tf.flags.DEFINE_string('summary_dir', './summary',
                       'The directory to write results.')
tf.flags.DEFINE_string('load_test_path', None, 
                       'The (processed) test set file to load.')

def default_hparams():
    """Builds an HParams object with default hperparameters."""
    return tf.contrib.training.HParams(
        decay_rate=0.96,
        decay_steps=2000,
        leaky=False,
        learning_rate=0.001,
        loss_type='margin',
        num_prime_capsules=32,
        padding='VALID',
        remake=True,
        routing=3,
        verbose=False)
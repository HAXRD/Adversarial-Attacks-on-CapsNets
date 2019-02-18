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

"""
1. Training
    - After converged, produce adversarial examples with
      test set and find out success rate of adversarial 
      attacks. Compare the model-method-dataset difference.
2. Adversarial training
    - use the same raw input, but process the input using 
      different methods before feeding into the model for 
      training, then find out success rate of adversarial 
      attacks.
    - use the white-box adversarial attack to see the 
      model resistance.
3. Transferability of adversarial attacks
            | CNN  |  Caps  | Caps+R |
    CNN     |  -   |        |        |
    Caps    |      |    -   |        |
    Caps+R  |      |        |    -   |
"""

from __future__ import absolute_import, division, print_function

import os 
import sys
import time
import re 
import numpy as np
import tensorflow as tf 
from glob import glob

import logging
import daiquiri

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

from config import FLAGS, default_hparams

from models import capsule_model
from models import cnn_model
MODELS = {
    'caps': capsule_model.CapsuleModel,
    'caps_r': capsule_model.CapsuleModel,
    'cnn': cnn_model.CNNModel
}

import dataset.dataset_utils as dataset_utils

import adversarial_noise

def find_event_file_path(load_dir):
    """
    Finds the event file. TODO: might remove it
    :param load_dir: the directory to look for the training 
    :return: path to event file
    """
    fpath_list = glob(os.path.join(load_dir, 'events.*'))
    if len(fpath_list) == 1:
        return fpath_list[0]
    else:
        raise ValueError

def find_latest_ckpt_info(load_dir, find_all=False):
    """
    Finds the latest checkpoint information.
    :param load_dir: the directory to look for the training ckpt files
    :return: 
        latest global step
        latest ckpt path
        step_ckpt pair list
    """
    ckpt = tf.train.get_checkpoint_state(load_dir)
    if ckpt and ckpt.model_checkpoint_path:
        latest_step = extract_step(ckpt.model_checkpoint_path)
        if find_all == True:
            ckpt_paths = glob(os.path.join(load_dir, 'model.ckpt-*.index'))
            pairs = [(int(re.search('\d+', os.path.basename(path)).group(0)), 
                      os.path.join(os.path.dirname(path), os.path.basename(path)[:-6]))
                      for path in ckpt_paths]
            pairs = sorted(pairs, key=lambda pair: pair[0])
        else:
            pairs = []
        return latest_step, ckpt.model_checkpoint_path, pairs
    return -1, None, []

def extract_step(path):
    """
    Returns the step from the file format name of Tensorflow checkpoints.
    :param path: the ckpt path returned by tf.train.get_checkpoint_state,
                 the format is {ckpt_name}-{step}
    :return: the last training step number of the ckpt
    """
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])

def run_train_session(iterator, specs, 
                      adversarial_method, epsilon, iteration_n,
                      write_dir, max_epochs, 
                      joined_result, save_epochs):
    """
    Start session to train the model.
    :param iterator: dataset iterator
    :param specs: a dictionary contains specifications
    :param adversarial_method: the adversarial training method
    :param epsilon: epsilon value 
    :param iteration_n: number of iteration for adversarial training procedure
    :param write_dir: summary directory to store ckpts
    :param max_epochs: maximum epochs to train the model
    :param joined_results: a JoinedResult namedtuple that stores the related tensors
    :param save_epochs: how often to store a ckpt
    """
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(write_dir, sess.graph)

        batch_data = iterator.get_next()
        sess.run(iterator.initializer)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        saver = tf.train.Saver(max_to_keep=None)

        epoch_time = 0
        total_time = 0
        step_counter = 0
        epochs_done = 0
        # Restore ckpt if not restart
        latest_step, latest_ckpt_fpath, _ = find_latest_ckpt_info(write_dir, False)
        if latest_step != -1 and latest_ckpt_fpath != None:
            saver.restore(sess, latest_ckpt_fpath)
            step_counter = latest_step
            epochs_done = step_counter // specs['steps_per_epoch']
        total_steps = specs['steps_per_epoch'] * (max_epochs - epochs_done)

        # Compute adversarial tensor
        if adversarial_method in ['BIM', 'ILLCM']:
            xs_advs = []
            for i in range(specs['num_gpus']):
                xs = tf.get_collection('tower_%d_batched_images' % i)[0]
                if adversarial_method == 'BIM':
                    loss = tf.get_collection('tower_%d_BIM_loss' % i)[0]
                elif adversarial_method == 'ILLCM':
                    loss = tf.get_collection('tower_%d_ILLCM_loss' % i)[0]
                # shape (100, 28, 28, 1)
                logger.info('Start computing gradients for tower_{}...'.format(i))
                xs_adv = adversarial_noise.compute_one_step_adv_avg(loss, xs, specs['batch_size'], epsilon)
                xs_advs.append(xs_adv) # [(100, 28, 28, 1), (100, 28, 28, 1)]

        # Start feeding process
        for stp in range(total_steps):
            start_anchor = time.time()
            step_counter += 1

            try:
                feed_dict = {}
                for i in range(specs['num_gpus']):
                    batch_val = sess.run(batch_data)
                    images = batch_val['images']
                    labels = batch_val['labels']
                    
                    x_t = tf.get_collection('tower_%d_batched_images' % i)[0]
                    y_t = tf.get_collection('tower_%d_batched_labels' % i)[0]

                    feed_dict[y_t] = labels
                    if adversarial_method in ['BIM', 'ILLCM'] and stp >= total_steps-(50)*specs['steps_per_epoch']:
                    # if adversarial_method in ['BIM', 'ILLCM']:
                        for j in range(iteration_n):
                            feed_dict[x_t] = images
                            images = sess.run(xs_advs[i], feed_dict=feed_dict)
                    else:
                        feed_dict[x_t] = images
                
                # Run inferences
                summary, accuracy, _ = sess.run(
                    [joined_result.summary, joined_result.accuracy, joined_result.train_op], 
                    feed_dict=feed_dict)
                
                # Add summary
                writer.add_summary(summary, global_step=step_counter)
                # Calculate time 
                time_consuming = time.time() - start_anchor
                epoch_time += time_consuming
                total_time += time_consuming
                # Save ckpts
                if step_counter % (specs['steps_per_epoch'] * save_epochs) == 0:
                    ckpt_path = saver.save(
                        sess, os.path.join(write_dir, 'model.ckpt'),
                        global_step=step_counter)
                    print("{0} epochs done (step = {1}), accuracy {2:.4f}. {3:.2f}s, checkpoint saved at {4}".format(
                        step_counter // specs['steps_per_epoch'], 
                        step_counter, 
                        accuracy, 
                        epoch_time, 
                        ckpt_path))
                    epoch_time = 0
                elif step_counter % specs['steps_per_epoch'] == 0:
                    print("{0} epochs done (step = {1}), accuracy {2:.4f}. {3:.2f}s".format(
                        step_counter // specs['steps_per_epoch'], 
                        step_counter, 
                        accuracy, 
                        epoch_time))
                    epoch_time = 0
                else:
                    print("running {0} epochs {1:.1f}%, total time ~ {2}:{3}:{4}".format(
                        step_counter // specs['steps_per_epoch'] + 1,
                        step_counter % specs['steps_per_epoch'] * 100.0 / specs['steps_per_epoch'],
                        int(total_time // 3600), 
                        int(total_time % 3600 // 60), 
                        int(total_time % 60)),
                        end='\r')
            except tf.errors.OutOfRangeError:
                break
            # Finished one step
        print('total time: {0}:{1}:{2}, accuracy: {3:.4f}.'.format(
            int(total_time // 3600), 
            int(total_time % 3600 // 60), 
            int(total_time % 60),
            accuracy))

def train(hparams, num_gpus, data_dir, dataset, 
                   adversarial_method, epsilon, iteration_n,
                   model_type, total_batch_size, image_size, 
                   summary_dir, save_epochs, max_epochs):
    """
    The function to train the model. One can specify the parameters 
    to train different models, datasets, with/without adversarial training.
    :param hparams: the hyperparameters to build the model graph
    :param num_gpus: number of GPUs to use
    :param data_dir: the directory contains the input data
    :param dataset: the name of the dataset for the experiment
    :param adversarial_method: method to generating adversarial examples
    :param epsilon: epsilon size for the adversarial precedure
    :param iteration_n: number of iterations to run the iterative process
    :param model_type: the abbreviation of model architecture type
    :param total_batch_size: total batch size that will be distributed to {num_gpus} GPUs
    :param image_size: image size after cropping/resizing
    :param summary_dir: the directory to write summaries and save the model
    :param save_epochs: how often the model save the ckpt file
    :param max_epochs: maximum epochs to train
    """
    if adversarial_method == 'Default':
        train_folder_name = 'train'
    else:
        train_folder_name = 'train_{}_eps{}_iter{}'.format(
            adversarial_method, epsilon, iteration_n)
    # Define subfolder in {summary_dir}, where we store the event and ckpt files.
    write_dir = os.path.join(summary_dir, model_type, dataset, adversarial_method, train_folder_name)
    # Define model graph
    with tf.Graph().as_default():
        distributed_dataset, specs = dataset_utils.inputs(
            dataset_name=dataset, total_batch_size=total_batch_size, num_gpus=num_gpus,
            max_epochs=max_epochs, resized_size=image_size, data_dir=data_dir,
            split='train')
        iterator = distributed_dataset.make_initializable_iterator()

        model = MODELS[model_type](hparams, specs)
        # build a model on multiple GPUs and returns a tuple of 
        # (a list of input tensor placeholders, a list of output
        # tensor placeholders)
        joined_result = model.build_model_on_multi_gpus()

        """Print stats"""
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        """"""

        run_train_session(iterator, specs, 
                          adversarial_method, epsilon, iteration_n,
                          write_dir, max_epochs, 
                          joined_result, save_epochs)

def run_test_session(iterator, specs, load_dir):
    """
    Load available ckpts and test accuracy.
    :param iterator: dataset iterator
    :param specs: a dictionary contains dataset specifications
    :param load_dir: the directory to store ckpt files
    """
    # Find latest step, ckpt, and all step-ckpt pairs
    latest_step, latest_ckpt_path, _ = find_latest_ckpt_info(load_dir, True)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('{0}\n ckpt files not fould!\n {0}'.format('='*20))
    else:
        print('{0}\nFound a ckpt!\n{0}'.format('='*20))
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        batch_data = iterator.get_next()

        saver.restore(sess, latest_ckpt_path)
        sess.run(iterator.initializer)

        acc_t = tf.get_collection('accuracy')[0]
        accs = []

        while True:
            try:
                flag = [False for i in range(specs['num_gpus'])]
                feed_dict = {}
                for i in range(specs['num_gpus']):
                    batch_val = sess.run(batch_data)
                    print(batch_val['images'].shape)
                    if specs['batch_size'] == batch_val['images'].shape[0]:
                        flag[i] = True
                        feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['images']
                        feed_dict[tf.get_collection('tower_%d_batched_labels' % i)[0]] = batch_val['labels']
                if all(flag):
                    acc = sess.run(acc_t, feed_dict=feed_dict)
                    accs.append(acc)
                    print("accuracy: {}".format(acc))
            except tf.errors.OutOfRangeError:
                break
        mean_acc = np.mean(accs)
        print("overall accuracy: {}".format(mean_acc))

def test(num_gpus, 
         adversarial_method, epsilon, iteration_n,
         total_batch_size, image_size,
         summary_dir,
         load_test_path):
    """
    The function to test the model. One can specify the parameters 
    to test different models, datasets, with/without adversarially
    trained model.
    :param num_gpus: number of GPUs available to use
    :param adversarial_method: this set of parameters determines the 
                               summary folder to load
    :param epsilon: epsilon size for the adversarial procedure
    :param iteration_n: number of iterations to run the iterative process
    :param total_batch_size: total batch size that will be distributed to {num_gpus} GPUs;
    :param image_size: image size after cropping/resizing
    :param summary_dir: the directory to write summaries and save the model
    :param load_test_path: test set path to load
    """
    if adversarial_method == 'Default':
        train_folder_name = 'train'
    else:
        train_folder_name = 'train_{}_eps{}_iter{}'.format(adversarial_method, epsilon, iteration_n)
    # Define path to ckpts
    load_dir = os.path.join(summary_dir, train_folder_name)
    assert os.path.exists(load_test_path) == True
    assert os.path.isfile(load_test_path) == True

    split = os.path.basename(load_test_path).split('.')[0]
    dataset = os.path.basename(os.path.dirname(load_test_path))
    data_dir = os.path.dirname(load_test_path)

    # Declare an empty model graph
    with tf.Graph().as_default():
        # get batched dataset and declare initializable iterator
        distributed_dataset, specs = dataset_utils.inputs(
            dataset_name=dataset, total_batch_size=total_batch_size, num_gpus=num_gpus,
            max_epochs=1, resized_size=image_size, data_dir=data_dir,
            split=split)
        iterator = distributed_dataset.make_initializable_iterator()
        run_test_session(iterator, specs, load_dir)

def run_gen_adv_session(iterator, specs, data_dir, load_dir, 
                        adversarial_method, epsilon, iteration_n, all_):
    """
    Start session to generate adversarial examples with given
    set and method parameters.
    :param iterator: dataset iterator
    :param specs: a dictionary that contains dataset specification
    :param data_dir: data directory to load .npz file 
    :param load_dir: summary directory to load ckpt files
    :param adversarial_method: method to generating adversarial examples
    :param epsilon: epsilon size for the adversarial procedure
    :param iteration_n: number of iterations to run the iterative process
    :param all_: an auxilary parameter for unitT
    """
    # Find latest step, ckpt, and all step-ckpt pairs
    latest_step, latest_ckpt_path, _ = find_latest_ckpt_info(load_dir, True)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('{0}\n ckpt files not fould!\n {0}'.format('='*20))
    else:
        print('{0}\nFound a ckpt!\n{0}'.format('='*20))
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        batch_data = iterator.get_next()

        saver.restore(sess, latest_ckpt_path)
        sess.run(iterator.initializer)

        adv_images = []
        adv_labels = []

        total_time = 0
        iteration_time = 0
        counter = 0

        if all_ == None:
            total_iteration = int(specs['total_size'] // specs['num_gpus'])
        else:
            total_iteration = all_
        
        # Compute adversarial tensor
        xs_advs = []
        for i in range(specs['num_gpus']):
            xs_split = [tf.get_collection('tower_%d_batched_images_split' % i)[j] for j in range(specs['batch_size'])]
            if adversarial_method == 'BIM':
                loss = tf.get_collection('tower_%d_BIM_loss' % i)[0]
            elif adversarial_method == 'ILLCM':
                loss = tf.get_collection('tower_%d_ILLCM_loss' % i)[0]
            # shape (100, 28, 28, 1)
            logger.info('Start computing gradients for tower_{}...'.format(i))
            xs_adv = adversarial_noise.compute_one_step_adv(loss, xs_split, specs['batch_size'], epsilon) 
            xs_advs.append(xs_adv) # [(100, 28, 28, 1), (100, 28, 28, 1)]
        
        logger.info("Start generating adversarial examples...")
        for _ in range(total_iteration):
            start_anchor = time.time()
            try:
                flag = [False for i in range(specs['num_gpus'])]
                for i in range(specs['num_gpus']):
                    feed_dict = {}
                    batch_val = sess.run(batch_data)
                    images = batch_val['images'] # (100, 28, 28, 1)
                    labels = batch_val['labels'] # (100,)
                    if images.shape[0] == specs['batch_size']:
                        flag[i] = True
                        x_t = tf.get_collection('tower_%d_batched_images' % i)[0]
                        y_t = tf.get_collection('tower_%d_batched_labels' % i)[0]

                        feed_dict[y_t] = labels
                        for j in range(iteration_n):
                            feed_dict[x_t] = images
                            images = sess.run(xs_advs[i], feed_dict=feed_dict)
                        
                        adv_images.append(images)
                        adv_labels.append(labels)
                if not all(flag):
                    for i in range(specs['num_gpus']):
                        adv_images.pop()
                        adv_labels.pop()

                counter += specs['num_gpus']*specs['batch_size']

                iteration_time = time.time() - start_anchor
                total_time += iteration_time

                print("Finished {0}% ~ {1}/{2}, iteration: {3:.3f}s, total: {4}:{5}:{6}".format(
                    100 * float(counter) / float(specs['total_size']),
                    counter, specs['total_size'],
                    iteration_time,
                    int(total_time // 3600),
                    int(total_time % 3600 // 60),
                    int(total_time % 60)))
            except tf.errors.OutOfRangeError:
                break
        adv_images = np.concatenate(adv_images, axis=0)
        adv_labels = np.concatenate(adv_labels, axis=0)
        adv_labels = np.argmax(adv_labels, axis=1)
        fname = 'test_{}_eps{}_iter{}.npz'.format(
            adversarial_method, int(epsilon), int(iteration_n))
        fpath = os.path.join(data_dir, fname)
        np.savez(fpath, x=adv_images, y=adv_labels)
        logger.info("{} saved to '{}'!".format(fname, fpath))
        
def gen_adv(num_gpus, data_dir, dataset,
            total_batch_size, image_size,
            summary_dir, 
            adversarial_method, epsilon=0.01, iteration_n=1, all_=None):
    """
    Generate adversarial examples with given set and method parameters.
    :param num_gpus: number of GPUs available to use 
    :param data_dir: data directory to load 
    :param dataset: dataset name 
    :param total_batch_size: total number of batch size, which will 
                             evenly divided into {num_gpus} partitions
    :param image_size: image size after resizing
    :param summary_dir: summary directory to load ckpt files
    :param adversarial_method: this set of parameters determines the 
                               summary folder to load
    :param epsilon: epsilon size for the adversarial procedure
    :param iteration_n: number of iterations to run the iterative process
    :param all_:an auxilary parameter for unitT
    """
    # define path to ckpts
    load_dir = os.path.join(summary_dir, 'train')
    # declare an empty model graph
    with tf.Graph().as_default():
        # get batched dataset object and declare initializable iterator
        distributed_dataset, specs = dataset_utils.inputs(
            dataset_name=dataset, total_batch_size=total_batch_size, num_gpus=num_gpus,
            max_epochs=1, resized_size=image_size, data_dir=data_dir,
            split='test')
        iterator = distributed_dataset.make_initializable_iterator()
        run_gen_adv_session(iterator, specs, data_dir, load_dir, adversarial_method, epsilon, iteration_n, all_)

def main(_):
    hparams = default_hparams()
    if FLAGS.hparams_override:
        hparams.parse(FLAGS.hparams_override)
    
    if FLAGS.mode == 'train':
        train(hparams, FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, 
                       FLAGS.adversarial_method, FLAGS.epsilon, FLAGS.iteration_n,
                       FLAGS.model, FLAGS.total_batch_size, FLAGS.image_size, 
                       FLAGS.summary_dir, FLAGS.save_epochs, FLAGS.max_epochs)
    elif FLAGS.mode == 'gen_adv':
        gen_adv(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset,
                FLAGS.total_batch_size, FLAGS.image_size,
                FLAGS.summary_dir, 
                FLAGS.adversarial_method, FLAGS.epsilon, FLAGS.iteration_n)
    elif FLAGS.mode == 'test':
        test(FLAGS.num_gpus, 
             FLAGS.adversarial_method, FLAGS.epsilon, FLAGS.iteration_n,
             FLAGS.total_batch_size, FLAGS.image_size,
             FLAGS.summary_dir,
             FLAGS.load_test_path)
    else:
        raise ValueError("No matching mode found for '{}'".format(FLAGS.mode))

if __name__ == '__main__':
    tf.app.run()
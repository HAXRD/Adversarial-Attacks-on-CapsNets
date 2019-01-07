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
    'cnn': cnn_model.CNNModel
}

import dataset.dataset_utils as dataset_utils

import adversarial_noise

def find_event_file_path(load_dir):
    """Finds the event file.

    Args:
        load_dir: the directory to look for the training checkpoints.
    Returns:
        path to the event file.
    """
    fpath_list = glob(os.path.join(load_dir, 'events.*'))
    if len(fpath_list) == 1:
        return fpath_list[0]
    else:
        raise ValueError

def find_latest_ckpt_info(load_dir, find_all=False):
    """Finds the latest checkpoint information.

    Args:
        load_dir: the directory to look for the training checkpoints.
    Returns:
        latest global step, latest checkpoint path, step_ckpt pair list
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
    """Returns the step from the file format name of Tensorflow checkpoints.

    Args:
        path: The checkpoint path returned by tf.train.get_checkpoint_state.
            The format is: {ckpnt_name}-{step}
    Returns:
        The last training step number of the checkpoint.
    """
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])

def run_train_session(iterator, specs, 
                      adversarial_method,
                      write_dir, max_epochs, 
                      joined_result, save_epochs):
    """Start session to train the model.

    Args:
        iterator: dataset iterator;
        specs: dict, dataset specifications;
        adversarial_method: str, adversarial training method;
        write_dir: summary directory to store the ckpts;
        max_epochs: maximum epochs to train the model;
        joined_results: a namedtuple that stores the related tensors;
        save_epochs: how many epochs to store a ckpt.
    """
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare summary writer and save the graph in the meanwhile
        writer = tf.summary.FileWriter(write_dir, sess.graph)
        # declare batched data instance and initialize the iterator
        batch_data = iterator.get_next()
        sess.run(iterator.initializer)
        # initialize variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # declare saver object for future saving
        saver = tf.train.Saver(max_to_keep=None)

        epoch_time = 0
        total_time = 0
        step_counter = 0
        epochs_done = 0
        # restore ckpt if not restart
        latest_step, latest_ckpt_fpath, _ = find_latest_ckpt_info(write_dir, False)
        if latest_step != -1 and latest_ckpt_fpath != None:
            saver.restore(sess, latest_ckpt_fpath)
            step_counter = latest_step
            epochs_done = step_counter // specs['steps_per_epoch']
        total_steps = specs['steps_per_epoch'] * (max_epochs - epochs_done)

        # start feeding process
        for _ in range(total_steps):
            start_anchor = time.time()
            step_counter += 1

            try:
                # get placeholders and create feed_dict
                feed_dict = {}
                for i in range(specs['num_gpus']):
                    batch_val = sess.run(batch_data)
                    if adversarial_method != None:
                        pass
                        # batch_val = ADVERSARIAL_METHOD[adversarial_method](batch_val, loss, sess)
                    feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['images']
                    feed_dict[tf.get_collection('tower_%d_batched_labels' % i)[0]] = batch_val['labels']
                
                """Run inferences"""
                summary, accuracy, _ = sess.run(
                    [joined_result.summary, joined_result.accuracy, joined_result.train_op], 
                    feed_dict=feed_dict)
                
                """Add summary"""
                writer.add_summary(summary, global_step=step_counter)
                # calculate time 
                time_consuming = time.time() - start_anchor
                epoch_time += time_consuming
                total_time += time_consuming
                """Save ckpts"""
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
                   adversarial_method,
                   model_type, total_batch_size, image_size, 
                   summary_dir, save_epochs, max_epochs):
    """The function to train the model. One can specify the parameters 
    to train different models, datasets, with/without adversarial training.

    Args:
        hparams: the hyperparameters to build the model graph;
        num_gpus: number of GPUs to use;
        data_dir: the directory containing the input data;
        dataset: the name of the dataset for the experiment;
        adversarial_method: method to generating adversarial examples;
        model_type: the abbreviation of model architecture;
        total_batch_size: total batch size, which will be distributed to {num_gpus} GPUs;
        image_size: image size after cropping/resizing;
        summary_dir: the directory to write summaries and save the model;
        save_epochs: how often the model save the ckpt;
        max_epochs: maximum epochs to train;
    """
    # define subfolder in {summary_dir}, where we store the event and ckpt files.
    write_dir = os.path.join(summary_dir, model_type, dataset, adversarial_method, 'train')
    # define model graph
    with tf.Graph().as_default():
        # get batched_dataset and declare initializable iterator
        distributed_dataset, specs = dataset_utils.inputs(
            dataset_name=dataset, total_batch_size=total_batch_size, num_gpus=num_gpus,
            max_epochs=max_epochs, resized_size=image_size, data_dir=data_dir,
            split='train')
        iterator = distributed_dataset.make_initializable_iterator()
        # initialize model with hparams and specs
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
                          adversarial_method,
                          write_dir, max_epochs, 
                          joined_result, save_epochs)

def run_test_session(iterator, specs, load_dir):
    """Load available ckpts and test accuracy.

    Args:
        iterator: dataset iterator;
        specs: dict, dataset specifications;
        load_dir: str, directory to store ckpts;
    """
    # find latest step, ckpt, and all step-ckpt pairs
    latest_step, latest_ckpt_path, _ = find_latest_ckpt_info(load_dir, True)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('{0}\n ckpt files not fould!\n {0}'.format('='*20))
    else:
        print('{0}\nFound a ckpt!\n{0}'.format('='*20))
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # import compute graph
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        # get dataset object working 
        batch_data = iterator.get_next()

        # restore variables
        saver.restore(sess, latest_ckpt_path)
        sess.run(iterator.initializer)

        acc_t = tf.get_collection('accuracy')[0]
        accs = []

        while True:
            try:
                # get placeholders and create feed dict
                feed_dict = {}
                for i in range(specs['num_gpus']):
                    batch_val = sess.run(batch_data)
                    feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['images']
                    feed_dict[tf.get_collection('tower_%d_batched_labels' % i)[0]] = batch_val['labels']
                acc = sess.run(acc_t, feed_dict=feed_dict)
                accs.append(acc)
                print("accuracy: {}".format(acc))
            except tf.errors.OutOfRangeError:
                break
        mean_acc = np.mean(accs)
        print("overall accuracy: {}".format(mean_acc))

def test(num_gpus, 
         total_batch_size, image_size,
         summary_dir,
         load_test_path):
    """The function to test the model. One can specify the parameters 
    to test different models, datasets, with/without adversarially
    trained model.
    
    Args:
        num_gpus: number of GPUs available to use;
        total_batch_size: total batch size, which will be distributed to {num_gpus} GPUs;
        image_size: image size after cropping/resizing;
        summary_dir: the directory to write summaries and save the model;
        load_test_path: test set path to load.
    """
    # define path to ckpts
    load_dir = os.path.join(summary_dir, 'train')
    # make sure target test file exists
    assert os.path.exists(load_test_path) == True
    assert os.path.isfile(load_test_path) == True

    split = os.path.basename(load_test_path).split('.')[0]
    dataset = os.path.basename(os.path.dirname(load_test_path))
    data_dir = os.path.dirname(load_test_path)

    # declare an empty model graph
    with tf.Graph().as_default():
        # get batched dataset and declare initializable iterator
        distributed_dataset, specs = dataset_utils.inputs(
            dataset_name=dataset, total_batch_size=total_batch_size, num_gpus=num_gpus,
            max_epochs=1, resized_size=image_size, data_dir=data_dir,
            split=split)
        iterator = distributed_dataset.make_initializable_iterator()
        run_test_session(iterator, specs, load_dir)

def run_gen_adv_session(iterator, specs, data_dir, load_dir, adversarial_method, eps, iteration_n, all_):

    # find latest step, ckpt, and all step-ckpt pairs
    latest_step, latest_ckpt_path, _ = find_latest_ckpt_info(load_dir, True)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('{0}\n ckpt files not fould!\n {0}'.format('='*20))
    else:
        print('{0}\nFound a ckpt!\n{0}'.format('='*20))
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # import compute graph
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        # get dataset object 
        batch_data = iterator.get_next()

        # restore variables
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
        
        # compute adversarial tensor
        xs_advs = []
        for i in range(specs['num_gpus']):
            xs_split = [tf.get_collection('tower_%d_batched_images_split' % i)[j] for j in range(specs['batch_size'])]
            if adversarial_method == 'BIM':
                loss = tf.get_collection('tower_%d_BIM_loss' % i)[0]
            elif adversarial_method == 'ILLCM':
                loss = tf.get_collection('tower_%d_ILLCM_loss' % i)[0]
            # shape (100, 28, 28, 1)
            xs_adv = adversarial_noise.compute_one_step_adv(loss, xs_split, specs['batch_size'], eps) 
            xs_advs.append(xs_adv) # [(100, 28, 28, 1), (100, 28, 28, 1)]
        
        for _ in range(total_iteration):
            start_anchor = time.time()
            try:
                for i in range(specs['num_gpus']):
                    feed_dict = {}
                    batch_val = sess.run(batch_data)
                    images = batch_val['images'] # (100, 28, 28, 1)
                    labels = batch_val['labels'] # (100,)

                    x_t = tf.get_collection('tower_%d_batched_images' % i)[0]
                    y_t = tf.get_collection('tower_%d_batched_labels' % i)[0]

                    feed_dict[y_t] = labels
                    for j in range(iteration_n):
                        feed_dict[x_t] = images
                        images = sess.run(xs_advs[i], feed_dict=feed_dict)
                    
                    adv_images.append(images)
                    adv_labels.append(labels)

                counter += specs['num_gpus']*specs['batch_size']

                iteration_time = time.time() - start_anchor
                total_time += iteration_time

                print("Finished {0}% ~ {1}/{2}, iteration: {3}s, total: {4}:{5}:{6}".format(
                    100 * float(counter) / float(specs['total_size']),
                    counter, specs['total_size'],
                    iteration_time,
                    int(total_time // 3600),
                    int(total_time % 3600 //60),
                    int(total_time % 3600)))
            except tf.errors.OutOfRangeError:
                break
        adv_images = np.concatenate(adv_images, axis=0)
        adv_images = np.squeeze(adv_images, axis=1)
        adv_labels = np.concatenate(adv_labels, axis=0)
        adv_labels = np.argmax(adv_labels, axis=1)
        fname = 'test_{}.npz'.format(adversarial_method)
        fpath = os.path.join(data_dir, fname)
        np.savez(fpath, x=adv_images, y=adv_labels)
        logger.info("{} saved to '{}'!".format(fname, fpath))
        
def gen_adv(num_gpus, data_dir, dataset,
            adversarial_method,
            total_batch_size, image_size,
            summary_dir, eps=0.01, iteration_n=1, all_=None):
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
        run_gen_adv_session(iterator, specs, data_dir, load_dir, adversarial_method, eps, iteration_n, all_)

def main(_):
    hparams = default_hparams()
    if FLAGS.hparams_override:
        hparams.parse(FLAGS.hparams_override)
    
    if FLAGS.mode == 'train':
        train(hparams, FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, 
                       FLAGS.adversarial_method,
                       FLAGS.model, FLAGS.total_batch_size, FLAGS.image_size, 
                       FLAGS.summary_dir, FLAGS.save_epochs, FLAGS.max_epochs)
    elif FLAGS.mode == 'gen_adv':
        gen_adv(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset,
                FLAGS.adversarial_method,
                FLAGS.total_batch_size, FLAGS.image_size,
                FLAGS.summary_dir, FLAGS.epsilon, FLAGS.iteration_n)
    elif FLAGS.mode == 'test':
        test(FLAGS.num_gpus, 
             FLAGS.total_batch_size, FLAGS.image_size,
             FLAGS.summary_dir,
             FLAGS.load_test_path)
    else:
        raise ValueError("No matching mode found for '{}'".format(FLAGS.mode))

if __name__ == '__main__':
    tf.app.run()
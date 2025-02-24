# Copyright 2015 Google Inc. All Rights Reserved.
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

# This file has been modified by Samuel Murray.

from __future__ import print_function
from __future__ import division

import time
import os
from datetime import datetime
import numpy as np
import tensorflow as tf

import conv_net as cnn
import utilities

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_string('train_dir', './pascal_train/', 'train dir')
flags.DEFINE_string('prep_train_dir', './preprocessed/training/', 'preprocessed training images')
flags.DEFINE_string('prep_val_dir', './preprocessed/validation/', 'preprocessed validation images')

# class ImagesHolder(object):
#     """docstring for ClassName"""
#     def __init__(self, training_images, training_labels, validation_images, validation_labels):
#         self.train_images = training_images
#         self.train_labels = training_labels
#         self.val_images = validation_images
#         self.val_labels = validation_labels
#         self.train_index = np.arange(len(self.train_labels))
#         self.val_index = np.arange(len(self.val_labels))
#         self.train_pos = 0
#         self.val_pos = 0
#         self.shuffle()
#         self.train_pos_max = len(self.train_labels)
#         self.val_pos_max = len(self.val_labels)

#     def shuffle_training(self):
#         np.random.shuffle(self.train_index)
#         self.train_pos = 0 

#     def shuffle_validation(self):
#         np.random.shuffle(self.val_index)
#         self.val_pos = 0]

#     def get_next_train_batch(self, image_type):
#         batch_size = int(flags.batch_size)
#         images = None
#         labels = None
#         need_more = False
#         remainder = None

#         end = self.train_pos + batch_size
#         if end > self.train_pos_max:
#             end = self.train_pos_max
#             remainder = self.train_pos_max - (self.train_pos + batch_size)
#             need_more = True
        
#         images = self.train_images[self.train_index[self.train_pos:end]]
#         self.train_pos = self.train_pos + batch_size

#         if need_more:
#             self.shuffle_training()
#             images = np.stack(ima)


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32)
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def sample_images_and_labels(images, labels, batch_size):
    # TODO:
    # Make it so that it samples new images all the time, e.g. by shuffling the lists the first time,
    # and then go through that list linearly
    """
    This function samples random images and corresponding labels from the given sets.
    Currently, there is not way to prevent the same images being sampled over and over again.
    """
    size = len(labels)
    indices = np.random.choice(range(size), batch_size, replace=False)
    return images[indices], labels[indices]


def fill_feed_dict(images, labels, images_placeholder, labels_placeholder, keep_prob, batch_size=FLAGS.batch_size, validation=False):
    """Fills the feed_dict for training the given step.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next [batch size] examples.
    images_feed, labels_feed = sample_images_and_labels(images, labels, batch_size)
    if validation:
        feed_dict = {
            images_placeholder: images_feed,
            labels_placeholder: labels_feed,
            keep_prob: 1.0
        }
    else:
        feed_dict = {
            images_placeholder: images_feed,
            labels_placeholder: labels_feed,
            keep_prob: FLAGS.dropout
        }
    return feed_dict


def do_eval(sess, eval_correct, images, labels, images_placeholder, labels_placeholder, keep_prob):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    num_examples = len(labels)
    steps_per_epoch = num_examples // FLAGS.batch_size
    for step in range(steps_per_epoch - 1):
        feed_dict = {
            images_placeholder: images[(step * FLAGS.batch_size):((step + 1) * FLAGS.batch_size)],
            labels_placeholder: labels[(step * FLAGS.batch_size):((step + 1) * FLAGS.batch_size)],
            keep_prob: 1.0
        }
        #feed_dict = fill_feed_dict(images, labels, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def do_eval_per_class(sess, eval_correct, images, labels, images_placeholder, labels_placeholder, keep_prob):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
    """
    for i in np.unique(labels):
        indices = np.where(labels == i)
        print(' Class {}:'.format(utilities.name_by_label[i]))
        do_eval(sess, eval_correct, images[indices], labels[indices], images_placeholder, labels_placeholder, keep_prob)


def save_image_output(images, image_type):
    """
    Iterate trough each training and validation image, send it through the pretrained model, and save the output
    :param images: list of training image paths
    :param image_type: should be 'training' or 'validation'
    """
    def get_file_name(file_path):
        # images/train/cropped_2007_000032/aeroplane_False_0_0_281_281.jpg
        return file_path[file_path.find("_")+1:-4].replace("/", "_") + ".npy"

    if image_type != "training" and image_type != "validation":
        print("Wrong kwargs!")
        return
    # if not tf.gfile.Exists(FLAGS.prep_train_dir):
    #     tf.gfile.MakeDirs(FLAGS.prep_train_dir)
    # if not tf.gfile.Exists(FLAGS.prep_val_dir):
    #     tf.gfile.MakeDirs(FLAGS.prep_val_dir)

    output = cnn.inference_to_save()

    sess = tf.Session()

    for i, image in enumerate(images):
        save_path = 'data/preprocessed/{}/'.format(image_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_save_path = '{}{}'.format(save_path, get_file_name(image))
        print("full save path ", full_save_path)
        if os.path.isfile(full_save_path):
            print("Skipping ", full_save_path)
            continue

        if not tf.gfile.Exists(image):
            tf.logging.fatal('File does not exist %s', image)
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        precomputed_value = sess.run(output, {'DecodeJpeg/contents:0': image_data})
        reshaped_value = np.squeeze(precomputed_value)

        np.save(open(full_save_path, 'wb'), reshaped_value)
        print(reshaped_value.shape)


def run_training(training_images, training_labels, validation_images, validation_labels):

    # Remove conflicting files and create directories
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Create placeholders
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    keep_prob = tf.placeholder(tf.float32)

    # Define the different Ops from the network
    logits = cnn.inference(images_placeholder, keep_prob)
    loss = cnn.loss(logits, labels_placeholder)
    train_op = cnn.training(loss, FLAGS.learning_rate)
    eval_correct = cnn.evaluation(logits, labels_placeholder)

    summary_op = tf.merge_all_summaries()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in range(FLAGS.max_steps):
        start_time = time.time()
        
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        feed_dict = fill_feed_dict(training_images, training_labels, images_placeholder, labels_placeholder, keep_prob, validation=False)

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        # Write the summaries and print an overview fairly often.
        if step % 10 == 0:
            # Print status to stdout.
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            print('%s: Step %d: loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                  % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

        if step % 10 == 0:
            # Update the events file.
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            #print("Summary writer flushed with string {} at step {}".format(summary_str, step))
            summary_writer.flush()

        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            print("Model saved in file: {}".format(checkpoint_path))
            # Evaluate against the training set.
            print('Training Data Eval:')
            do_eval(sess, eval_correct, training_images, training_labels, images_placeholder, labels_placeholder, keep_prob)

            # Evaluate against the validation set.
            print('Validation Data Eval:')
            print(' Overall score:')
            do_eval(sess, eval_correct, validation_images, validation_labels, images_placeholder, labels_placeholder, keep_prob)

        if step + 1 == FLAGS.max_steps:
            do_eval_per_class(sess, eval_correct, validation_images, validation_labels,
                              images_placeholder, labels_placeholder, keep_prob)

    print("learning rate: ", flags.learning_rate)
    print("dropout: ", flags.dropout)
    print("max steps: ", flags.max_steps)
    print("batch size: ", flags.batch_size)


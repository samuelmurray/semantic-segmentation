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

import time
import os
from datetime import datetime
import numpy as np
import tensorflow as tf

import conv_net as cnn

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 5, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('train_dir', './pascal_train/', 'train dir')


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


def fill_feed_dict(images, labels, images_placeholder, labels_placeholder, batch_size=FLAGS.batch_size):
    """Fills the feed_dict for training the given step.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next [batch size] examples.
    images_feed, labels_feed = sample_images_and_labels(images, labels, batch_size)
    feed_dict = {
        images_placeholder: images_feed,
        labels_placeholder: labels_feed,
    }
    return feed_dict


def do_eval(sess, eval_correct, images, labels, images_placeholder, labels_placeholder):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    num_examples = len(labels)
    steps_per_epoch = num_examples // FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(images, labels, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training(training_images, training_labels, validation_images, validation_labels):

    # Remove conflicting files and create directories
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Create placeholders
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    # Define the different Ops from the network
    logits = cnn.inference(images_placeholder)
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
        feed_dict = fill_feed_dict(training_images, training_labels, images_placeholder, labels_placeholder)

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

        if step % 100 == 0:
            # Update the events file.
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            print("Summary writer flushed with string {} at step {}".format(summary_str, step))
            summary_writer.flush()

        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            print("Model saved in file: {}".format(checkpoint_path))
            # Evaluate against the training set.
            print('Training Data Eval:')
            do_eval(sess, eval_correct, training_images, training_labels, images_placeholder, labels_placeholder)

            # Evaluate against the validation set.
            print('Validation Data Eval:')
            do_eval(sess, eval_correct, validation_images, validation_labels, images_placeholder, labels_placeholder)

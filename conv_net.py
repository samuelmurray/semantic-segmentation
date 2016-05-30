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

# This file has been modified by Samuel Murray to use and modify an existing graph_def.
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
tf.app.flags.DEFINE_string(
    'model_dir', './imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

NUM_CLASSES = 21
TRAINABLE_VARIABLES = []


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        from six.moves import urllib

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def inference_to_save():
    """ Build the Pascal model up to where it may be used for inference.
    Args:
        resized_images: Input tensor with batch of images of size 299x299x3
    Returns:
        logits: Output tensor with the computed logits.
    """

    # Creates graph from saved graph_def.pb.
    maybe_download_and_extract()
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    # Use the second-last layer
    pool_tensor = tf.get_default_graph().get_tensor_by_name('pool_3:0')
    return pool_tensor


def inference(images_data, keep_prob):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # First FC layer, known as FCa
    with tf.name_scope("FCa") as scope:
        W_FCa = weight_variable([2048, 1024])
        b_FCa = bias_variable([1024])
        #pool_flat = tf.reshape(images_data, [-1, 2048])
        h_FCa = tf.nn.relu(tf.matmul(images_data, W_FCa) + b_FCa)
        h_FCa = tf.nn.dropout(h_FCa, keep_prob)

    # Readout layer, known as FCb
    with tf.name_scope("FCb") as scope:
        W_FCb = weight_variable([1024, NUM_CLASSES])
        b_FCb = bias_variable([NUM_CLASSES])
        logits = tf.matmul(h_FCa, W_FCb) + b_FCb

    TRAINABLE_VARIABLES.extend([W_FCa, b_FCa, W_FCb, b_FCb])
    return logits


def old_inference(resized_images):
    """ Build the Pascal model up to where it may be used for inference.
    Args:
        resized_images: Input tensor with batch of images of size 299x299x3
    Returns:
        logits: Output tensor with the computed logits.
    """

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Creates graph from saved graph_def.pb.
    maybe_download_and_extract()
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        input_map = {'Sub:0': resized_images}
        _ = tf.import_graph_def(graph_def, name='', input_map=input_map)

    # Use the second-last layer
    pool_tensor = tf.get_default_graph().get_tensor_by_name('pool_3:0')

    # First FC layer, known as FCa
    with tf.name_scope("FCa") as scope:
        W_FCa = weight_variable([2048, 1024])
        b_FCa = bias_variable([1024])
        pool_flat = tf.reshape(pool_tensor, [-1, 2048])
        h_FCa = tf.nn.relu(tf.matmul(pool_flat, W_FCa) + b_FCa)

    # Readout layer, known as FCb
    with tf.name_scope("FCb") as scope:
        W_FCb = weight_variable([1024, NUM_CLASSES])
        b_FCb = bias_variable([NUM_CLASSES])
        logits = tf.matmul(h_FCa, W_FCb) + b_FCb

    TRAINABLE_VARIABLES.extend([W_FCa, b_FCa, W_FCb, b_FCb])
    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
    Returns:
        loss: Loss tensor of type float.
    """
    with tf.name_scope("Loss") as scope:
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """
    with tf.name_scope("Training") as scope:
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(loss.op.name, loss)
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=TRAINABLE_VARIABLES)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    with tf.name_scope("Evaluation") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))

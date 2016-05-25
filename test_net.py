import time

import numpy as np
import tensorflow as tf

import segmentation_net as cnn

batch_size = 1
learning_rate = 0.01

image = 'imagenet/panda.jpeg'
if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
image_data = tf.gfile.FastGFile(image, 'rb').read()
image_label = np.zeros(shape=(batch_size))

logits = cnn.inference()
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
loss = cnn.loss(logits, labels_placeholder)
train_op = cnn.training(loss, 0.01)
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
summary_writer = tf.train.SummaryWriter('tmp', sess.graph)

for step in range(10):
    start_time = time.time()
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    feed_dict = {'DecodeJpeg/contents:0': image_data,
                 labels_placeholder: image_label}
    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

    duration = time.time() - start_time

    # Write the summaries and print an overview fairly often.
    # Print status to stdout.
    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
    # Update the events file.
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)
    summary_writer.flush()

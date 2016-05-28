from __future__ import print_function
from __future__ import division

import numpy as np
import conv_net_feed as cnnfeed
import conv_net_util as cnnutil
from utilities import training_images, training_labels, validation_images, validation_labels

# Define the data
"""
image = 'imagenet/panda.jpeg'
image2 = 'imagenet/panda-update.jpg'
image_names = [image, image2, image]
image_labels = np.array([0, 6, 0])  # Make sure to use the same label for identical images
image_data = cnnutil.preprocess_images(image_names)
cnnfeed.run_training(image_data, image_labels, image_data, image_lables)
"""
how_many = 1000

train_names = training_images[0:how_many]
train_labels = training_labels[0:how_many]
val_names = validation_images[0:10]
val_labels = validation_labels[0:10]

# Preprocess data
train_data = cnnutil.preprocess_images(train_names)
val_data = cnnutil.preprocess_images(val_names)

cnnfeed.run_training(train_data, train_labels, val_data, val_labels)

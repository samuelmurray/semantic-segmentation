from __future__ import print_function
from __future__ import division

import numpy as np
import conv_net_feed as cnnfeed
import conv_net_util as cnnutil
from utilities import training_images, training_labels, validation_images, validation_labels

# Define the data
image = 'imagenet/panda.jpeg'
image2 = 'imagenet/panda-update.jpg'
image_names = [image, image2, image]
batch_size = len(image_names)

image_names = training_images[0:100]
image_labels = training_labels[0:100]


# Preprocess data
image_data = cnnutil.preprocess_images(image_names)
image_labels = np.array([0, 6, 0])  # Make sure to use the same label for identical images

cnnfeed.run_training(image_data, image_labels, validation_images[0:10], validation_labels[0:10])

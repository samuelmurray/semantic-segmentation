from __future__ import print_function
from __future__ import division

import numpy as np
import time
import conv_net_feed as cnnfeed
import conv_net_util as cnnutil
from utilities import training_images, training_labels, validation_images, validation_labels
# from utilities import training_images_small, training_labels_small, validation_images_small, validation_labels_small

# Define the data
"""
image = 'imagenet/panda.jpeg'
image2 = 'imagenet/panda-update.jpg'
image_names = [image, image2, image]
image_labels = np.array([0, 6, 0])  # Make sure to use the same label for identical images
image_data = cnnutil.preprocess_images(image_names)
cnnfeed.run_training(image_data, image_labels, image_data, image_lables)
"""
how_many_train = 100  # set to large number to take all
how_many_val = 0  # set to large number to take all

train_names = training_images[0:how_many_train]
train_labels = training_labels[0:how_many_train]
val_names = validation_images[0:how_many_val]
val_labels = validation_labels[0:how_many_val]

start_time = time.time()
cnnfeed.save_image_output(train_names, image_type="training")
cnnfeed.save_image_output(val_names, image_type="validation")
print("Saving {} train and {} val images took {}s".format(how_many_train, how_many_val, time.time() - start_time))

# # Preprocess data
# train_data = cnnutil.preprocess_images(train_names)
# val_data = cnnutil.preprocess_images(val_names)
#
# cnnfeed.run_training(train_data, train_labels, val_data, val_labels)

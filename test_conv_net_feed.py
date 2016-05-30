from __future__ import print_function
from __future__ import division

import os
import pickle
import numpy as np
import time
import conv_net_feed as cnnfeed
import conv_net_util as cnnutil
import utilities
#from utilities import training_images, training_labels, validation_images, validation_labels
# from utilities import training_images_small, training_labels_small, validation_images_small, validation_labels_small

# Define the data
"""
image = 'imagenet/panda.jpeg'
image2 = 'imagenet/panda-update.jpg'
image_names = [image, image2, image]
image_labels = np.array([0, 6, 0])  # Make sure to use the same label for identical images
image_data = cnnutil.preprocess_images(image_names)
cnnfeed.run_training(image_data, image_labels, image_data, image_labels)
"""


def get_images_and_label(image_type):
    if image_type != 'training' and image_type != 'validation':
        print("Wrong image_type argument. Got \'{}\', expected \'training\' or \'validation\'".format(image_type))
        return
    data_dir = 'data/preprocessed/{}/'.format(image_type)

    images = []
    labels = []
    for i, file in enumerate(os.listdir(data_dir)):
        if i % 1000 == 0:
            print("Loaded {} of the {} images".format(i, image_type))
        if file.endswith(".npy"):
            split_name = file.split('_')
            if len(split_name) < 2:
                continue
            images.append(np.load(open(data_dir + file, 'rb')))
            label = utilities.label_by_name[split_name[2]]
            labels.append(label)
    labels = np.array(labels)
    images = np.stack(images)
    return images, labels


train_images, train_labels = get_images_and_label('training')
np.save(open('data/preprocessed/train_images.npy', 'wb'), train_images)
np.save(open('data/preprocessed/train_labels.npy', 'wb'), train_labels)

val_images, val_labels = get_images_and_label('validation')
np.save(open('data/preprocessed/val_images.npy', 'wb'), val_images)
np.save(open('data/preprocessed/val_labels.npy', 'wb'), val_labels)

print("Training images size: \t %s MB" % (train_images.nbytes/1024/1024))
print("Training labels size: \t %s MB" % (train_labels.nbytes/1024/1024))
print("Validation images size: \t %s MB" % (val_images.nbytes/1024/1024))
print("Validation labels size: \t %s MB" % (val_labels.nbytes/1024/1024))

exit()
cnnfeed.run_training(train_images, train_labels, val_images, val_labels)


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
"""

# # Preprocess data
# train_data = cnnutil.preprocess_images(train_names)
# val_data = cnnutil.preprocess_images(val_names)
#
# cnnfeed.run_training(train_data, train_labels, val_data, val_labels)

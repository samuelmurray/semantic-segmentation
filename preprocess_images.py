from __future__ import print_function
from __future__ import division

import numpy as np
import time
import conv_net_feed as cnnfeed
import conv_net_util as cnnutil
#from utilities import training_images, training_labels, validation_images, validation_labels
from utilities import train_images_by_label, val_images_by_label
#from utilities import name_by_label, label_by_name
import argparse
import random

# from utilities import training_images_small, training_labels_small, validation_images_small, validation_labels_small
chosen_labels = ["car", "cow", "train", "cat", "person", "background"]


def parse_args():
    parser = argparse.ArgumentParser(description="Process images.")
    parser.add_argument("-i", "--index", help="index of label to process")
    args = parser.parse_args()
    return args


args = parse_args()
index = int(args.index)
label = chosen_labels[index]

train_names = train_images_by_label[label]
val_names = val_images_by_label[label]

random.shuffle(train_names)
random.shuffle(val_names)

train_names = train_names[0:3000]
val_names = val_names[0:1000]


start_time = time.time()
cnnfeed.save_image_output(train_names, image_type="training")
cnnfeed.save_image_output(val_names, image_type="validation")
print("Saving {} train and {} val images took {}s".format(len(train_names), len(val_names), time.time() - start_time))

# # Preprocess data
# train_data = cnnutil.preprocess_images(train_names)
# val_data = cnnutil.preprocess_images(val_names)
#
# cnnfeed.run_training(train_data, train_labels, val_data, val_labels)
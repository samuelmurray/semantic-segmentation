#
# Running this will iterate through the generated test and validation images in
# * images/test
# * images/validate
#
# and save the files detailed below
#  * label_by_name: label name --> label number
#  * name_by_label: label number --> label name
#  * training_images: list of paths to training images
#  * validation_images: lists of paths to validation images
#  * training_labels: numpy vector of float32
#  * valication_labels: numpy vector of float32

from __future__ import print_function
from __future__ import division
import os
from collections import OrderedDict
import pickle
from numpy import zeros
import numpy as np
from collections import defaultdict


def save_labels():
    label_by_image = {}  # image --> label number   NOT SAVED
    val_images_by_label = defaultdict(list)  # label --> [image, image, ...]
    train_images_by_label = defaultdict(list)  # label --> [image, image, ...]
    all_labels = set()  # NOT SAVED
    one_hot_label_dictionary = OrderedDict()  # label --> one-hot vector  NOT SAVED (currenly not used)
    label_by_name = {}  # label name --> label number
    name_by_label = {}  # label number --> label name
    training_images = []  # list of training image paths
    validation_images = []  # list of validation image paths
    training_images_small = []  # skip background and hard
    validation_images_small = [] # skip background and hard



    # Figure out the mapping between image and label
    for root, dirs, files in os.walk('images/'):
        if len(files) > 1:  # somtimes only .DS file
            # file_names = [name for name in files if (name != '.DS') and ('invalid' not in name)]
            short_file_names = [name for name in files if (name != '.DS') and ('invalid' not in name)]
            file_names = [os.path.join(root, name) for name in short_file_names]
            labels = [name[0:name.find('_')] for name in short_file_names]
            label_by_image.update(dict(zip(file_names, labels)))
            all_labels.update(labels)

            for i, image in enumerate(file_names):
                label = labels[i]
                if 'train' in root:
                    train_images_by_label[label].append(image)
                else:
                    val_images_by_label[label].append(image)


            if 'train' in root:
                training_images = training_images + file_names
                for image in file_names:
                    if not ("background" in image or "_yes_" in image):  # skip background and difficult
                        training_images_small.append(image)
            else:
                validation_images = validation_images + file_names
                for image in file_names:
                    if not ("background" in image or "_yes_" in image):  # skip background and difficult
                        validation_images_small.append(image)

    # build the label dictionary
    num_labels = len(all_labels)
    for i, label in enumerate(sorted(all_labels)):
        vec = zeros(num_labels)
        vec[i] = 1.0
        one_hot_label_dictionary[label] = vec
        label_by_name[label] = i
        name_by_label[i] = label

    training_labels = np.zeros(len(training_images), dtype=np.float32)
    for i, image in enumerate(training_images):
        label = label_by_image[image]
        training_labels[i] = label_by_name[label]

    validation_labels = np.zeros(len(validation_images), dtype=np.float32)
    for i, image in enumerate(validation_images):
        label = label_by_image[image]
        validation_labels[i] = label_by_name[label]

    # save everything to disk
    pickle.dump(label_by_name, open('data/pickles/label_by_name.p', 'wb'))
    pickle.dump(name_by_label, open('data/pickles/name_by_label.p', 'wb'))
    #pickle.dump(label_by_image, open('data/pickles/label_by_image.p', 'wb'))
    pickle.dump(validation_images, open('data/pickles/validation_images.p', 'wb'))
    pickle.dump(training_images, open('data/pickles/training_images.p', 'wb'))
    #pickle.dump(training_labels, open('data/pickles/training_labels.p', 'wb'))
    #pickle.dump(validation_labels, open('data/pickles/validation_labels.p', 'wb'))
    #pickle.dump(one_hot_label_dictionary, open('data/pickles/one_hot_label_dictionary.p', 'wb'))
    np.save(open('data/pickles/training_labels.npy', 'wb'), training_labels)
    np.save(open('data/pickles/validation_labels.npy', 'wb'), validation_labels)

    pickle.dump(train_images_by_label, open('data/pickles/train_images_by_label.p', 'wb'))
    pickle.dump(val_images_by_label, open('data/pickles/val_images_by_label.p', 'wb'))

    # # no background or hard images
    # # build the label dictionary
    # all_labels_small = all_labels.copy()
    # all_labels_small.remove('background')
    # label_by_name_small = {}
    # name_by_label_small = {}
    # for i, label in enumerate(sorted(all_labels_small)):
    #     label_by_name_small[label] = i
    #     name_by_label_small[i] = label
    #
    # training_labels_small = np.zeros(len(training_images_small), dtype=np.float32)
    # for i, image in enumerate(training_images_small):
    #     label = label_by_image[image]
    #     training_labels_small[i] = label_by_name_small[label]
    #
    # validation_labels_small = np.zeros(len(validation_images_small), dtype=np.float32)
    # for i, image in enumerate(validation_images_small):
    #     label = label_by_image[image]
    #     validation_labels_small[i] = label_by_name_small[label]
    #
    # pickle.dump(validation_images_small, open('data/pickles/validation_images_small.p', 'wb'))
    # pickle.dump(training_images_small, open('data/pickles/training_images_small.p', 'wb'))
    # np.save(open('data/pickles/training_labels_small.npy', 'wb'), training_labels_small)
    # np.save(open('data/pickles/validation_labels_small.npy', 'wb'), validation_labels_small)


if __name__ == "__main__":
    save_labels()

# images/train/cropped_2007_000032/images/train/cropped_2007_000032/aeroplane_False_0_0_281_281.jpg
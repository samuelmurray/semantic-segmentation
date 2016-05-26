# Intended to store utilites we will need during our run.
# At present those are:
#
# * label_by_image: Full path to image file -> label of image (e.g. 'airplane')
# * label_dictionary: label -> one-hot vector
# * training_images: list of full path to training training images
# * validation_images: list of fll path to validation images

import pickle
import os
import numpy as np

paths = {
    'label_by_name': 'data/pickles/label_by_name.p',
    'name_by_label': 'data/pickles/name_by_label.p',
    'validation_images': 'data/pickles/validation_images.p',
    'training_images': 'data/pickles/training_images.p',
    'training_labels': 'data/pickles/training_labels.npy',
    'validation_labels': 'data/pickles/validation_labels.npy'
}


if not all(os.path.isfile(path) for path in paths.values()):
    print("Generating image labels")
    from save_image_labels import save_labels
    save_labels()


label_by_name = pickle.load(open(paths['label_by_name'], 'rb'))
name_by_label = pickle.load(open(paths['name_by_label'], 'rb'))
training_images = pickle.load(open(paths['training_images'], 'rb'))
training_labels = np.load(open(paths['training_labels'], 'rb'))
validation_images = pickle.load(open(paths['validation_images'], 'rb'))
validation_labels = np.load(open(paths['validation_labels'], 'rb'))

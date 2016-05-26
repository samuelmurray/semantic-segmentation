# Intended to store utilites we will need during our run.
# At  present those are:
#
# * label_by_image: Full path to image file -> label of image (e.g. 'airplane')
# * label_dictionary: label -> one-hot vector
# * training_images: list of full path to training training images
# * validation_images: list of fll path to validation images

import pickle
import os

LABEL_BY_IMAGE_PICKLE = 'data/pickles/label_by_image.p'
LABEL_DICTIONARY_PICKLE = 'data/pickles/label_dictionary.p'
TRAINING_IMAGES_PICKLE = 'data/pickles/training_images.p'
VALIDATION_IMAGES_PICKLE = 'data/pickles/validation_images.p'

if not (os.path.isfile(LABEL_BY_IMAGE_PICKLE) and \
        os.path.isfile(LABEL_DICTIONARY_PICKLE) and \
        os.path.isfile(TRAINING_IMAGES_PICKLE) and \
        os.path.isfile(VALIDATION_IMAGES_PICKLE)):
    print("Generating image labels")
    from save_image_labels import save_labels
    save_labels()


label_by_image = pickle.load(open('data/pickles/label_by_image.p', 'rb'))
label_dictionary = pickle.load(open('data/pickles/label_dictionary.p', 'rb'))
training_images = pickle.load(open('data/pickles/training_images.p', 'rb'))
validation_images = pickle.load(open('data/pickles/validation_images.p', 'rb'))




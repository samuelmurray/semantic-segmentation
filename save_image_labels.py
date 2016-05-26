#
# Running this will iterate through the generated test and validation images in
# * images/test
# * images/validate
#
# and generate two dictionaries for lookup:
#
# * label_by_image: image name --> label,
#     * e.g. background_False_87_145_377_435.jpg --> background
# * label_dictionary: label --> one-hot vector as a numpy array
#     * e.g. aeroplane --> array([ 1.,  0.,  0.,  0.,  ...     ])


import os
from collections import OrderedDict
import pickle
from numpy import zeros


def save_labels():
    label_by_image = {}
    all_labels = set()
    label_dictionary = OrderedDict()
    training_images = []
    validation_images = []

    # Figure out the mapping between image and label
    for root, dirs, files in os.walk('images/'):
        if len(files) > 1:  # somtimes only .DS file
            # file_names = [name for name in files if (name != '.DS') and ('invalid' not in name)]
            short_file_names = [name for name in files if (name != '.DS') and ('invalid' not in name)]
            file_names = [os.path.join(root, name) for name in files if (name != '.DS') and ('invalid' not in name)]
            labels = [name[0:name.find('_')] for name in short_file_names]
            label_by_image.update(dict(zip(file_names, labels)))
            all_labels.update(labels)

            if 'train' in root:
                training_images = training_images + [os.path.join(root, name) for name in file_names]
            else:
                validation_images = validation_images + [os.path.join(root, name) for name in file_names]

    # build the label dictionary
    num_labels = len(all_labels)
    for i, label in enumerate(sorted(all_labels)):
        vec = zeros(num_labels)
        vec[i] = 1.0
        label_dictionary[label] = vec

    # save everything to disk
    pickle.dump(label_dictionary, open('data/pickles/label_dictionary.p', 'wb'))
    pickle.dump(label_by_image, open('data/pickles/label_by_image.p', 'wb'))
    pickle.dump(validation_images, open('data/pickles/validation_images.p', 'wb'))
    pickle.dump(training_images, open('data/pickles/training_images.p', 'wb'))

if __name__ == "__main__":
    save_labels()
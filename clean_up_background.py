from __future__ import print_function
from __future__ import division

import os
import random
import numpy as np
import shutil

data_dir = data_dir = 'data/preprocessed/training/'

background_files = []

counter = 0
for file_name in os.listdir(data_dir):
    if file_name.endswith(".npy"):
        split_name = file_name.split('_')
        if len(split_name) < 2:
            continue
        label = split_name[2]
        if label == "background":
            background_files.append(file_name)

file_index = [i for i in range(len(background_files))]
random.shuffle(file_index)
files_to_keep = file_index[0:9999]
files_to_skip = file_index[10000:]

for file_name in [background_files[i] for i in files_to_keep]:
    pass

old_path = 'data/preprocessed/training/%s'
new_path = 'data/preprocessed/moved_background/%s'
for file_name in [background_files[i] for i in files_to_skip]:
    shutil.move(old_path % file_name, new_path % file_name)




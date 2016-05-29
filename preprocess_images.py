from __future__ import print_function
from __future__ import division
from utilities import training_images, validation_images

from functools import partial
from multiprocessing.pool import Pool
from time import time
import glob

TRAIN_PATH = "images/train/*/*.jpg"
VALIDATE_PATH = "images/validate/*/*.jpg"

def get_file_name(file_path):

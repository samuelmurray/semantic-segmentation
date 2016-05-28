# Code to generate and optionally save crops of images

from __future__ import print_function
from __future__ import division
from typing import Iterator
from ImageCrop import ImageCrop



def generate_crops(image, max_num_images = 500) :
    lambdas = [1, 1.3, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4]
    width, height = image.size
    count = 0
    for l in lambdas:
        s = min(width, height) / l
        i = 0
        within_width = True
        while within_width:
            j = 0
            within_height = True
            while within_height:
                x_min = round(i * s)
                y_min = round(j * s)
                x_max = round((i + 1) * s)
                y_max = round((j + 1) * s)
                if x_max > width:
                    x_max = width
                    x_min = round(width - s)
                    within_width = False
                if y_max > height:
                    y_max = height
                    y_min = round(height - s)
                    within_height = False
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                yield ImageCrop(cropped_image, x_min, y_min, x_max, y_max)
                j += 0.5 # Increase i and j by 0.5 to get 50% overlap between neighbouring crops
                count += 1
                if count >= max_num_images:
                    print("The maximum number of images ({}) was generated".format(max_num_images))
                    return
            i += 0.5
    print("{} images were generated and saved".format(count))


def generate_and_save_crops(image, image_name, save_location = None, max_num_images = 500):
    import os
    if save_location is None:
        save_location = "images/cropped_{}/".format(image_name)
    if not os.path.isdir(save_location):
        os.makedirs(save_location)
    for image_crop in generate_crops(image, max_num_images):
        image_crop.save(save_location)

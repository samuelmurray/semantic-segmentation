#
# This spawns 4 processes that will:
# * iterate through all PASCAL images
# * generate all the crops
# * check what label each crop should get
# * save a cropped image with the label in folder images/train or images/validate as:
#     label_difficult_xmin_ymin_xmax_ymax

from functools import partial
from multiprocessing.pool import Pool
from time import time
from typing import Iterator
from PIL import Image
from PascalImage import PascalImage
from ImageCrop import ImageCrop
import os

JPEG_PATH = 'data/VOC2012/JPEGImages/%s.jpg'


def generate_crops(image_name, max_num_images: int = 500) -> Iterator[ImageCrop]:
    image = Image.open(JPEG_PATH % image_name)
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
                j += 0.5  # Increase i and j by 0.5 to get 50% overlap between neighbouring crops
                count += 1
                if count >= max_num_images:
                    print("The maximum number of images ({}) was generated".format(max_num_images))
                    return
            i += 0.5
    print("{} images were generated and saved".format(count))


def process_image(image_name, ts, vs,):
    save_location = None
    if image_name in ts:
        save_location = "images/train/cropped_{}/".format(image_name)
    else:
        save_location = "images/validate/cropped_{}/".format(image_name)

    if not os.path.isdir(save_location):
        os.makedirs(save_location)

    pi = PascalImage(image_name)
    for crop in generate_crops(image_name):
        label = pi.calculate_patch_type(crop)
        crop.save(save_location, "%s_%s" % (label[0], label[1]))


def main():
    ts = time()
    train_file_path = 'data/VOC2012/ImageSets/Segmentation/train.txt'
    val_file_path = 'data/VOC2012/ImageSets/Segmentation/val.txt'

    train_files = [l.strip() for l in open(train_file_path).readlines()]
    val_files = [l.strip() for l in open(val_file_path).readlines()]

    train_val_files = train_files + val_files

    train_set = set(train_files)
    val_set = set(val_files)

    process = partial(process_image, ts=train_set, vs=val_set)
    print(process)

    with Pool(4) as p:
        p.map(process, train_val_files)

    print('Took {}s'.format(time() - ts))

if __name__ == "__main__":
    main()


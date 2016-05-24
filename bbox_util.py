import xmltodict
import glob
import os
from typing import List
import pickle

from BoundingBox import BoundingBox
from PascalImage import PascalImage


def get_image_name(file_path: str) -> str:
    start = len(file_path) - file_path[::-1].find("/")
    end = len(file_path) - file_path[::-1].find(".") - 1
    return file_path[start:end] + '.jpg'


def xml_to_dict(xml_file_path: str, xml_attribs: bool=True):
    with open(xml_file_path, "rb") as f:  # notice the "rb" mode
        d = xmltodict.parse(f, xml_attribs=xml_attribs)
        return d


def generate_bounding_box(data) -> BoundingBox:
    object_name = data['name']
    is_difficult = bool(int(data['difficult']))
    coords = dict([(key, val) for key, val in data['bndbox'].items()])
    x_min = int(coords['xmin'])
    x_max = int(coords['xmax'])
    y_min = int(float(coords['ymin']))  # One rouge image had a float coordinate
    y_max = int(coords['ymax'])
    return BoundingBox(object_name, x_min, y_min, x_max, y_max, is_difficult)


def bounding_boxes_of_image(image_path: str) -> List[BoundingBox]:
    data = xml_to_dict(image_path)

    data = data['annotation']['object']
    ret = []

    if isinstance(data, dict):
        ret.append(generate_bounding_box(data))
    else:
        for obj in data:
            ret.append(generate_bounding_box(obj))
    return ret


def get_bounding_boxes(folder_path: str):
    file_path = 'data/pickles/bounding_boxes_by_image.p'
    if os.path.isfile(file_path):
        print("reading bounding boxes from file")
        return pickle.load(open(file_path, 'rb'))
    g = glob.iglob(folder_path + '*.xml')
    bounding_boxes_by_image = {}
    # counter = 0
    for file_name in g:
        image_name = get_image_name(file_name)
        bounding_boxes = bounding_boxes_of_image(file_name)
        bounding_boxes_by_image[image_name] = PascalImage(image_name, bounding_boxes)
        # counter += 1
        # if counter > 30:
        #     break
    pickle.dump(bounding_boxes_by_image, open(file_path, 'wb'))
    return bounding_boxes_by_image


bounding_boxes = get_bounding_boxes('data/VOC2012/Annotations/')

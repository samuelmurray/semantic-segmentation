import xmltodict
import glob


def short_file_name(file_path):
    start = len(file_path) - file_path[::-1].find("/")
    end = len(file_path) - file_path[::-1].find(".") - 1
    return file_path[start:end]


def xml_to_dict(xml_file_path, xml_attribs=True):
    with open(xml_file_path, "rb") as f:    # notice the "rb" mode
        d = xmltodict.parse(f, xml_attribs=xml_attribs)
        return d


def bounding_box_objects(file_path):
    picture_name = short_file_name(file_path) + '.jpg'
    data = xml_to_dict(file_path)

    data = data['annotation']['object']
    ret = []

    if isinstance(data, dict):
        object_name = data['name']
        coords = dict([(key, val) for key,val in data['bndbox'].items()])
        ret.append({object_name: coords})
        return picture_name, ret

    for i, obj in enumerate(data):
        object_name = obj['name']
        coords = dict([(key, val) for key,val in obj['bndbox'].items()])
        ret.append({object_name: coords})
    return picture_name, ret


def bounding_boxes():
    g = glob.iglob('../data/VOC2012/Annotations/*.xml')
    bounding_boxes = {}
    for file_name in g:
        picture_name, data = bounding_box_objects(file_name)
        bounding_boxes[picture_name] = data
    return bounding_boxes

bounding_boxes = bounding_boxes()
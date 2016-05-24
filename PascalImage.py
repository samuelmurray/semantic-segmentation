from BoundingBox import BoundingBox
from ImageCrop import ImageCrop
import xmltodict


class PascalImage:

    def __init__(self, image_name: str):
        self.XML_FOLDER_PATH = 'data/VOC2012/Annotations/%s.xml'
        self.IMG_FOLDER_PATH = 'data/VOC2012/JPEGImages/%s.jpg'
        self.name = self.get_image_name(self.IMG_FOLDER_PATH % image_name)
        self.bounding_boxes = []
        self.bounding_boxes_of_image(self.IMG_FOLDER_PATH % image_name)

    def get_image_name(self, file_path):
        start = len(file_path) - file_path[::-1].find("/")
        end = len(file_path) - file_path[::-1].find(".") - 1
        return file_path[start:end]

    def xml_to_dict(self, image_path, xml_attribs=True):
        with open(self.XML_FOLDER_PATH % self.get_image_name(image_path), "rb") as f:  # notice the "rb" mode
            d = xmltodict.parse(f, xml_attribs=xml_attribs)
            return d

    def generate_bounding_box(self, data):
        object_name = data['name']
        is_difficult = bool(int(data['difficult']))
        coords = dict([(key, val) for key, val in data['bndbox'].items()])
        x_min = int(coords['xmin'])
        x_max = int(coords['xmax'])
        y_min = int(float(coords['ymin']))  # One rouge image had a float coordinate
        y_max = int(coords['ymax'])
        bbox = BoundingBox(object_name, x_min, y_min, x_max, y_max, is_difficult)
        self.bounding_boxes.append(bbox)

    def bounding_boxes_of_image(self, image_path):
        data = self.xml_to_dict(image_path)

        data = data['annotation']['object']

        if isinstance(data, dict):
            self.generate_bounding_box(data)
        else:
            for obj in data:
                self.generate_bounding_box(obj)

    def calculate_overlap(self, bbox: BoundingBox, patch: ImageCrop):
        x_overlap = max(0, min(bbox.x_max, patch.x_max) - max(bbox.x_min, patch.x_min))
        y_overlap = max(0, min(bbox.y_max, patch.y_max) - max(bbox.y_min, patch.y_min))
        return x_overlap * y_overlap

    def is_legal_overlap(self, bbox: BoundingBox, patch: ImageCrop, overlap: int):
        """
        The overlap is legal if the overlap is at least 20% of the patch area AND 60% of the bounding box area
        """
        # TODO: Change to instead of returning the name of the object, it returns the ID or one-hot numpy vector
        if (overlap >= 0.2 * patch.area) and overlap >= 0.6 * bbox.area:
            return True
        return False

    def calculate_patch_type(self, patch: ImageCrop):
        """
        Iterate through all of the patches for this image and check if the given patch satisfies
        the criteria to be included. If a patch overlaps more than one bounding box then we cannot
        use it
        :param patch: a cropped patch of the given image
        :return: what kind of patch this should be: {object_name, delete, background}
        """
        # TODO: Change to instead of returning the name of the object, it returns the ID or one-hot numpy vector
        overlap_count = 0
        label = None
        isDifficult = []
        for bbox in self.bounding_boxes:
            overlap = self.calculate_overlap(bbox, patch)
            if (overlap > 0) and self.is_legal_overlap(bbox, patch, overlap):
                if overlap_count == 1:
                    label = 'delete'
                    break
                overlap_count += 1
                label = bbox.name
                isDifficult.append(bbox.is_difficult)
        if overlap_count > 1:  # For a legal patch it can only overlap one bounding box
            return None, None  # Interpret this as 'delete' the patch
        else:
            return label, isDifficult  #

    def __str__(self):
        return "%s: \n\t%s" % (self.name, "\n\t".join([str(bbox) for bbox in self.bounding_boxes]))

    __repr__ = __str__

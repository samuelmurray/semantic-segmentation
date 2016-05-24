from typing import List

from BoundingBox import BoundingBox
from ImageCrop import ImageCrop


class PascalImage:

    def __init__(self, name: str, bounding_boxes: List[BoundingBox]):
        self.name = name
        self.bounding_boxes = bounding_boxes

    def add_bounding_box(self, bounding_box: BoundingBox):
        self.bounding_boxes.append(bounding_box)

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
            return labels, isDifficult  #

    def __str__(self):
        return "%s: \n\t%s" % (self.name, "\n\t".join([str(bbox) for bbox in self.bounding_boxes]))

    __repr__ = __str__






class ImageCrop:
    """
    Represents a crop of an image, with stored values of the coordinates in the original image.
    Used to compare the crop to bounding boxes of objects in the original image.
    """
    def __init__(self, image, x_min: int, y_min: int, x_max: int, y_max: int):
        self.image = image
        self.x_min = x_min  # type: int
        self.y_min = y_min  # type: int
        self.x_max = x_max  # type: int
        self.y_max = y_max  # type: int
class BoundingBox:
    __slots__ = ['name', 'x_min', 'y_min', 'x_max', 'y_max', 'is_difficult', 'area']

    def __init__(self, name: str, x_min: int, y_min: int, x_max: int, y_max: int, is_difficult: bool = False):
        self.name = name
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.is_difficult = is_difficult
        self.area = (x_max - x_min) * (y_max - y_min)

    def __str__(self):
        return "{name} ({difficult}): ({xmin},{ymin},{xmax},{ymax}), {area}px^2".format(
            name=self.name,
            difficult='difficult' if self.is_difficult else 'easy',
            xmin=self.x_min, ymin=self.y_min,
            xmax=self.x_max, ymax=self.y_max,
            area=self.area
        )

    __repr__ = __str__

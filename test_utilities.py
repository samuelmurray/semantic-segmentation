from pprint import pprint
from queue import Queue
from threading import Thread

from bbox_util import bounding_boxes


class ImageWorker(Thread):
    def __init__(self, queue, bboxes):
        Thread.__init__(self)
        self.queue = queue
        self.bounding_boxes = bboxes

    def run(self):
        while True:
            # Get work from the queue
            image_path = self.queue.get()
            # generate all patches
            # save along with label saying if they are valid
            self.queue.task_done()






from bbox_util import get_image_name
# from PIL import Image
# from crop_util import generate_and_save_crops
#
# image_path1 = 'data/VOC2012/JPEGImages/2007_000027.jpg'
# image_path2 = 'data/VOC2012/JPEGImages/2007_000032.jpg'
#
# image1 = Image.open(image_path1)
# image2 = Image.open(image_path2)
#
# generate_and_save_crops(image1, get_image_name(image_path1))
# generate_and_save_crops(image1, get_image_name(image_path2))

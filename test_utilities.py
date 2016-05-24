from functools import partial
from multiprocessing.pool import Pool
from PascalImage import PascalImage


image_path1 = '2007_000027'
pi = PascalImage(image_path1)
print(pi)

image_path2 = '2007_000032'
pi = PascalImage(image_path2)
print(pi)



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

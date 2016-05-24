from PIL import Image
from crop_util import *

image_folder = './imagenet/'
image_name = 'panda'
image_decoding = '.jpeg'
image = Image.open(image_folder + image_name + image_decoding)
print(type(image))
width, height = image.size
print("w = {}, h = {}".format(width, height))

max_images = 30
generate_and_save_crops(image, image_name, max_num_images=max_images)
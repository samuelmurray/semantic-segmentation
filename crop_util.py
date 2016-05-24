from PIL import Image
from ImageCrop import ImageCrop


def generate_crops(image, max_num_images: int = 500) -> ImageCrop:
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
                j += 0.5 # Increase i and j by 0.5 to get 50% overlap between neighbouring crops
                count += 1
                if count >= max_num_images:
                    print("The maximum number of images ({}) was generated".format(max_num_images))
                    return
            i += 0.5
    print("{} images were generated and saved".format(count))


def generate_and_save_crops(image, image_name: str, save_location: str = None, max_num_images: int = 500):
    import os
    if save_location is None:
        try:
            os.mkdir("images/")
        except: pass
        save_location = "images/cropped_{}/".format(image_name)
        dir = os.path.dirname(save_location)
        print(dir)
        try:
            os.mkdir(dir)
        except: pass
    for image_crop in generate_crops(image, max_num_images):
        #image_crop.image.save(save_location + "{}_{}_{}_{}".format(
        #    image_crop.x_min, image_crop.y_min, image_crop.x_max, image_crop.y_max) + ".jpg")
        image_crop.save(save_location)


def generate_crops2(image_path, max_num_images: int = 500):
    lambdas = [1, 1.3, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4]
    image = Image.open(image_path)
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
                # x_min: int, y_min: int, x_max: int, y_max: int
                bla = ImageCrop(image, x_min, y_min, x_max, y_max)
                yield bla
                j += 0.5
                count += 1
                if count >= max_num_images:
                    print("The maximum number of images ({}) was generated".format(max_num_images))
                    return
            i += 0.5
    print("{} patches were generated".format(count))

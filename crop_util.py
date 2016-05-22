def generate_crops(image, max_num_images: int = 500):
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
                start_x = round(i * s)
                start_y = round(j * s)
                end_x = round((i + 1) * s)
                end_y = round((j + 1) * s)
                if end_x > width:
                    end_x = width
                    start_x = round(width - s)
                    within_width = False
                if end_y > height:
                    end_y = height
                    start_y = round(height - s)
                    within_height = False
                cropped_image = image.crop((start_x, start_y, end_x, end_y))
                yield [cropped_image, start_x, start_y, end_x, end_y]
                j += 0.5
                count += 1
                if count >= max_num_images:
                    print("The maximum number of images ({}) was generated".format(max_num_images))
                    return
            i += 0.5
    print("{} images were generated and saved".format(count))


def generate_and_save_crops(image, image_name: str, save_location = None, max_num_images: int = 500):
    import os
    if save_location is None:
        try:
            os.mkdir("./images/")
        except: pass
        save_location = "./images/cropped_{}/".format(image_name)
        dir = os.path.dirname(save_location)
        print(dir)
        try:
            os.mkdir(dir)
        except: pass
    for item in generate_crops(image, max_num_images):
        item[0].save(save_location + "{}_{}_{}_{}".format(item[1], item[2], item[3], item[4]) + ".jpg")
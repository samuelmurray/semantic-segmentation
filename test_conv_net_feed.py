import conv_net_feed as cnnfeed
import numpy as np
import conv_net_util as cnnutil

# Define the data
image = 'imagenet/panda.jpeg'
image2 = 'imagenet/panda-update.jpg'
image_names = [image, image2, image]
batch_size = len(image_names)

# Preprocess data
image_data = cnnutil.preprocess_images(image_names)
image_labels = np.array([0, 6, 0])  # Make sure to use the same label for identical images

cnnfeed.run_training(image_data, image_labels, image_data, image_labels)

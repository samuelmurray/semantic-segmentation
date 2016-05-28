#
# Utility functions needed to use feed the network.
# At present there are only preprocessing functions, used to process images to the types the network expects


import tensorflow as tf


def preprocess_images(image_names):
    """
    :param image_names: list of image names to be preprocessed
    :return: numpy array of images resized to 299x299x3
    """
    images_data = []
    for image_name in image_names:
        if not tf.gfile.Exists(image_name):
            tf.logging.fatal('File does not exist %s', image_name)
        image_data = tf.gfile.FastGFile(image_name, 'rb').read()
        images_data.append(image_data)
    return preprocess_images_data(images_data)


def preprocess_images_data(images_data):
    """
    :param images_data: list of image data
    :return: numpy array of images resized to 299x299x3
    """
    with tf.name_scope("Pre-processing") as scope:
        resized_images = None
        for image_data in images_data:
            # TODO: not sure how this works for png files - would be nice if you could specify it somehow.
            decoded_jpeg = tf.image.decode_jpeg(image_data)
            # Seems we don't have to do this, as resize_bilinear can take a uint8 and output float32
            # casted = tf.cast(decoded_jpeg, dtype=tf.float64)
            decoded_jpeg = tf.expand_dims(decoded_jpeg, 0)
            resized_image = tf.image.resize_bilinear(decoded_jpeg, [299, 299])
            if resized_images is None:
                resized_images = resized_image
            else:
                resized_images = tf.concat(0, [resized_images, resized_image])

        preprocess_sess = tf.Session()
        resized_images = preprocess_sess.run(resized_images)
    return resized_images

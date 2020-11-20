"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import random
from glob import glob

import cv2
import numpy as np
import tensorflow as tf


class Image_data:
    def __init__(self, img_size, channels, dataset_path, augment_flag, resize_method='area'):
        self.img_height = img_size
        self.img_width = img_size
        self.channels = channels
        self.augment_flag = augment_flag

        self.dataset_path = dataset_path

        self.mask_images = []
        self.mask_masks = []
        self.nomask_images = []
        self.nomask_images2 = []
        self.nomask_masks = []
        self.nomask_masks2 = []
        self.resize_method = resize_method

    def image_processing(self, mask_path, mask_mask_path, nomask_path, nomask_mask_path, nomask_path2,
                         nomask_mask_path2):
        def make_getter(idx: int) -> callable:
            def _inner(a: np.ndarray) -> np.ndarray:
                return a[idx]

            return _inner

        getters = {key: make_getter(idx) for idx, key in enumerate('tlhw')}

        x = tf.io.read_file(mask_path)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        mask_image = tf.image.resize(x_decode, [self.img_height, self.img_width], method=self.resize_method)
        mask_image = preprocess_fit_train_image(mask_image)

        x = tf.io.read_file(mask_mask_path)
        # x_decode = tf.image.decode_png(x, channels=1)
        # mask_mask = tf.image.resize(x_decode, [self.img_height, self.img_width],
        #                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x_decode = tf.io.decode_raw(x, out_type=tf.int32)
        h = tf.numpy_function(getters['h'], [x_decode], tf.int32)
        w = tf.numpy_function(getters['w'], [x_decode], tf.int32)
        t = tf.numpy_function(getters['t'], [x_decode], tf.int32)
        l = tf.numpy_function(getters['l'], [x_decode], tf.int32)
        h = tf.clip_by_value(t + h, 1, self.img_height) - t
        w = tf.clip_by_value(l + w, 1, self.img_width) - l
        mask_mask = tf.ones([h, w, 1], dtype=tf.uint8) * 255
        mask_mask = tf.image.pad_to_bounding_box(mask_mask, t, l, self.img_height, self.img_width)

        x = tf.io.read_file(nomask_path)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        nomask_image = tf.image.resize(x_decode, [self.img_height, self.img_width], method=self.resize_method)
        nomask_image = preprocess_fit_train_image(nomask_image)


        x = tf.io.read_file(nomask_mask_path)
        x_decode = tf.io.decode_raw(x, out_type=tf.int32)
        h = tf.numpy_function(getters['h'], [x_decode], tf.int32)
        w = tf.numpy_function(getters['w'], [x_decode], tf.int32)
        t = tf.numpy_function(getters['t'], [x_decode], tf.int32)
        l = tf.numpy_function(getters['l'], [x_decode], tf.int32)
        h = tf.clip_by_value(t + h, 1, self.img_height) - t
        w = tf.clip_by_value(l + w, 1, self.img_width) - l
        nomask_mask = tf.ones([h, w, 1], dtype=tf.uint8) * 255
        nomask_mask = tf.image.pad_to_bounding_box(nomask_mask, t, l, self.img_height, self.img_width)

        x = tf.io.read_file(nomask_path2)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        nomask_image2 = tf.image.resize(x_decode, [self.img_height, self.img_width], method=self.resize_method)
        nomask_image2 = preprocess_fit_train_image(nomask_image2)

        x = tf.io.read_file(nomask_mask_path2)
        x_decode = tf.io.decode_raw(x, out_type=tf.int32)
        h = tf.numpy_function(getters['h'], [x_decode], tf.int32)
        w = tf.numpy_function(getters['w'], [x_decode], tf.int32)
        t = tf.numpy_function(getters['t'], [x_decode], tf.int32)
        l = tf.numpy_function(getters['l'], [x_decode], tf.int32)
        h = tf.clip_by_value(t + h, 1, self.img_height) - t
        w = tf.clip_by_value(l + w, 1, self.img_width) - l
        nomask_mask2 = tf.ones([h, w, 1], dtype=tf.uint8) * 255
        nomask_mask2 = tf.image.pad_to_bounding_box(nomask_mask2, t, l, self.img_height, self.img_width)

        if self.augment_flag:
            seed = random.randint(0, 2 ** 31 - 1)
            condition = tf.greater_equal(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), 0.5)

            augment_height_size = self.img_height + (30 if self.img_height == 256 else int(self.img_height * 0.1))
            augment_width_size = self.img_width + (30 if self.img_width == 256 else int(self.img_width * 0.1))

            mask_image, mask_mask = tf.cond(pred=condition,
                                            true_fn=lambda: augmentation(mask_image, mask_mask, augment_height_size,
                                                                         augment_width_size,
                                                                         seed, self.resize_method),
                                            false_fn=lambda: (mask_image, mask_mask))

            nomask_image, nomask_mask = tf.cond(pred=condition,
                                                true_fn=lambda: augmentation(nomask_image, nomask_mask,
                                                                             augment_height_size,
                                                                             augment_width_size,
                                                                             seed, self.resize_method),
                                                false_fn=lambda: (nomask_image, nomask_mask))

            nomask_image2, nomask_mask2 = tf.cond(pred=condition,
                                                  true_fn=lambda: augmentation(nomask_image2, nomask_mask2,
                                                                               augment_height_size,
                                                                               augment_width_size,
                                                                               seed, self.resize_method),
                                                  false_fn=lambda: (nomask_image2, nomask_mask2))

        mask_mask = tf.cast(mask_mask, dtype=tf.bool)  # Pixels for mask become True. Others become False.
        mask_mask = tf.squeeze(mask_mask)

        nomask_mask = tf.cast(nomask_mask, dtype=tf.bool)  # Pixels for mask become True. Others become False.
        nomask_mask = tf.squeeze(nomask_mask)

        nomask_mask2 = tf.cast(nomask_mask2, dtype=tf.bool)  # Pixels for mask become True. Others become False.
        nomask_mask2 = tf.squeeze(nomask_mask2)

        return mask_image, mask_mask, nomask_image, nomask_mask, nomask_image2, nomask_mask2

    def preprocess(self):
        mask_image_list = glob(os.path.join(self.dataset_path, 'mask') + '/*.png') + glob(
            os.path.join(self.dataset_path, 'mask') + '/*.jpg')

        # mask_mask_list = glob(os.path.join(self.dataset_path, 'mask_mask') + '/*.png') + glob(
        #     os.path.join(self.dataset_path, 'mask_mask') + '/*.jpg')
        mask_mask_list = [os.path.join(self.dataset_path, 'mask_mask', os.path.basename(m)) for m in mask_image_list]

        nomask_image_list = glob(os.path.join(self.dataset_path, 'nomask') + '/*.png') + glob(
            os.path.join(self.dataset_path, 'nomask') + '/*.jpg')

        # nomask_mask_list = glob(os.path.join(self.dataset_path, 'nomask_mask') + '/*.png') + glob(
        #     os.path.join(self.dataset_path, 'nomask_mask') + '/*.jpg')
        nomask_mask_list = [os.path.join(self.dataset_path, 'nomask_mask', os.path.basename(m)) for m in nomask_image_list]

        num_images = min(len(mask_image_list), len(nomask_image_list))
        mask_image_list = mask_image_list[:num_images]
        mask_mask_list = mask_mask_list[:num_images]
        nomask_image_list = nomask_image_list[:num_images]
        nomask_mask_list = nomask_mask_list[:num_images]

        nomask_image_list2 = random.sample(nomask_image_list, len(nomask_image_list))
        nomask_mask_list2 = [os.path.join(self.dataset_path, 'nomask_mask', os.path.basename(m)) for m in nomask_image_list2]

        self.mask_images.extend(mask_image_list)
        self.mask_masks.extend(mask_mask_list)
        self.nomask_images.extend(nomask_image_list)
        self.nomask_images2.extend(nomask_image_list2)
        self.nomask_masks.extend(nomask_mask_list)
        self.nomask_masks2.extend(nomask_mask_list2)


def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images


def preprocess_fit_train_image(images):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    return images


def postprocess_images(images):
    images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
    images = tf.cast(images, dtype=tf.dtypes.uint8)
    return images


def load_images(image_path, img_size, img_channel):
    x = tf.io.read_file(image_path)
    x_decode = tf.image.decode_jpeg(x, channels=img_channel, dct_method='INTEGER_ACCURATE')
    img = tf.image.resize(x_decode, [img_size, img_size])
    img = preprocess_fit_train_image(img)

    return img


def augmentation(image, mask, augment_height, augment_width, seed, resize_method='area'):
    if mask is not None:
        image = tf.concat([image, tf.cast(mask, dtype=tf.float32)], axis=-1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize(image, [augment_height, augment_width], method=resize_method)
    image = tf.image.random_crop(image, ori_image_shape, seed=seed)

    if mask is not None:
        # ori_mask_shape = tf.shape(mask)
        # mask = tf.image.random_flip_left_right(mask, seed=seed)
        # mask = tf.image.resize(mask, [augment_height, augment_width], method=resize_method)
        # mask = tf.image.random_crop(mask, ori_mask_shape, seed=seed)
        mask = image[..., 3:]
        image = image[..., :3]
        mask = tf.cast(mask, dtype=tf.uint8)

    image = tf.image.random_brightness(image, max_delta=0.09, seed=seed)
    # image = tf.image.random_contrast(image, lower=0., upper=0.2, seed=seed)
    return image, mask


def load_test_image(image_path, img_width, img_height, img_channel):
    if img_channel == 1:
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(img_width, img_height))

    if img_channel == 1:
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = np.expand_dims(img, axis=0)

    img = img / 127.5 - 1

    return img


def save_images(images, size, image_path):
    # size = [height, width]
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return ((images + 1.) / 2) * 255.0


def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h * j:h * (j + 1), w * i:w * (i + 1), :] = image

    return img


def return_images(images, size):
    x = merge(images, size)

    return x


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


def pytorch_xavier_weight_factor(gain=0.02, uniform=False):
    factor = gain * gain
    mode = 'fan_avg'

    return factor, mode, uniform


def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu'):
    if activation_function == 'relu':
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu':
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function == 'tanh':
        gain = 5.0 / 3
    else:
        gain = 1.0

    factor = gain * gain
    mode = 'fan_in'

    return factor, mode


def automatic_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def multiple_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
                 tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
from tensorflow.contrib.slim.nets import inception

import MobileNet as mn


class InceptionV3Model(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        probs = output.op.inputs[0]
        return probs


class MobilenetModel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        reuse = True if self.built else None
        with slim.arg_scope(mn.mobilenet_v1_arg_scope()):
            _, endpoints = mn.mobilenet_v1(inputs=x_input, num_classes=self.num_classes, dropout_keep_prob=1.0,
                                           is_training=False, reuse=reuse)
        self.built = True
        output = endpoints['Predictions']
        probs = output.op.inputs[0]
        return probs


def get_flags(method="fgsm"):
    tf.flags.DEFINE_integer("batch_size", 32, 'How many images to be processed at one time')
    tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')
    tf.flags.DEFINE_string(
        'checkpoint_path',
        r'D:\Dropbox\Code\AdversarialExampleRestoration\pre-trained\mobilenet_v1_1.0_224_2017_06_14',
        'Path to checkpoint for inception network.')
    tf.flags.DEFINE_string(
        'input_dir',
        r'C:\Datasets\Kaggle\nips2017_adversarial_attack\nips-2017-adversarial-learning-development-set\images',
        'Input directory with images.')
    tf.flags.DEFINE_string(
        'output_dir', 'E:/output', 'Output directory with images.')
    tf.flags.DEFINE_string("model_name", "inceptionv3", "name of aimed model")
    if method == "fgsm":
        tf.flags.DEFINE_float("eps", 0.2, "epsilon [0,2]")
        tf.flags.DEFINE_float("clip_min", -1.0, "clip min")
        tf.flags.DEFINE_float("clip_max", 1.0, "clip max")
        tf.flags.DEFINE_string("attack_method", "fgsm", "attack methods")
    if method == "evaluate":
        tf.flags.DEFINE_string(
            'adv_example_dir',
            r'E:/output',
            'path to adversarial examples')

    return tf.flags.FLAGS


def read_single_image(filepath, image_size):
    with tf.gfile.Open(filepath, mode="rb") as f:
        image = Image.open(f).convert('RGB')
        image = image.resize([image_size, image_size])
        image = np.array(image).astype(np.float) / 255.0
        return image * 2.0 - 1.0


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        images[idx, :, :, :] = read_single_image(filepath, batch_shape[1])
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'wb') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def get_image_size(name):
    if name == "inceptionv3":
        return 299
    if name == "mobilenet":
        return 224

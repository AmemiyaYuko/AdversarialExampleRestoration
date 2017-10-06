import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
from cleverhans.attacks import FastGradientMethod

import MobileNet as mn

tf.flags.DEFINE_integer("batch_size", 32, 'How many images to be processed at one time')
tf.flags.DEFINE_integer("image_size", 224, 'Size of images')
tf.flags.DEFINE_string(
    'checkpoint_path', 'ckpt/',
    'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir',
    'C:/Datasets/Kaggle/nips2017_adversarial_attack/nips-2017-adversarial-learning-development-set/images',
    'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', 'E:/output', 'Output directory with images.')
tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_float("eps", 0.2, "epsilon")
tf.flags.DEFINE_float("clip_min", -1.0, "clip min")
tf.flags.DEFINE_float("clip_max", 1.0, "clip max")
tf.flags.DEFINE_string("attack_method", "fgsm", "attack methods")
FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, mode="rb") as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
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
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'wb') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def fgsm_attack(prediction, images, eps=0.2, clip_min=-1.0, clip_max=1.0):
    fgsm = FastGradientMethod(prediction)
    adv = fgsm.generate(images, eps=eps, clip_min=clip_min, clip_max=clip_max)
    return adv


def generate(method):
    inputs = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])
    with slim.arg_scope(mn.mobilenet_v1_arg_scope()):
        _, endpoints = mn.mobilenet_v1(inputs=inputs, is_training=False, )
    prediction = endpoints['Predictions']
    if method == "fgsm":
        return fgsm_attack(prediction, inputs, eps=FLAGS.eps, clip_min=FLAGS.clip_min, clip_max=FLAGS.clip_max)


def main(_):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    num_classes = 1001
    tf.logging.set_verbosity(tf.logging.INFO)
    x_adv = generate(FLAGS.attack_method)
    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)

    print()


if __name__ == '__main__':
    tf.app.run()

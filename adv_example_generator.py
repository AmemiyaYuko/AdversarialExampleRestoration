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
    'master', '', 'The address of the TensorFlow master to use.')
tf.flags.DEFINE_string(
    'checkpoint_path',
    r'E:\Dropbox\Dropbox\Code\AdversarialExampleRestoration\pre-trained\mobilenet_v1_1.0_224_2017_06_14',
    'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir',
    'E:\Datasets\Kaggle\images',
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


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, mode="rb") as f:
            image=Image.open(f).convert('RGB')
            image=image.resize([FLAGS.image_size,FLAGS.image_size])
            image = np.array(image).astype(np.float) / 255.0

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
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'wb') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def fgsm_attack(model, images, eps=0.2, clip_min=-1.0, clip_max=1.0):
    fgsm = FastGradientMethod(model)
    adv = fgsm.generate(images, eps=eps, clip_min=clip_min, clip_max=clip_max)
    return adv


def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, shape=batch_shape)
        model = MobilenetModel(1001)
        x_adv = fgsm_attack(model, inputs, eps=FLAGS.eps, clip_min=FLAGS.clip_min, clip_max=FLAGS.clip_max)
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_dir=FLAGS.checkpoint_path, master=FLAGS.master)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={inputs: images})
                save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()

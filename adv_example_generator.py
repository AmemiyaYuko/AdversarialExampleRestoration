import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from cleverhans.attacks import FastGradientMethod

from utils import load_images, save_images, MobilenetModel, get_flags, InceptionV3Model, get_image_size

FLAGS = get_flags("fgsm")


def adv_attack(model, images, eps, clip_min, clip_max, method="fgsm"):
    if method == "fgsm":
        att = FastGradientMethod(model)
        return att.generate(images, eps=eps, clip_min=clip_min, clip_max=clip_max)
    return None


def model_selector(name="mobilenet"):
    if name == "mobilenet":
        return MobilenetModel(1001)
    if name == "inceptionv3":
        return InceptionV3Model(1001)
    return None


def main(_):
    image_size = get_image_size(FLAGS.model_name)
    batch_shape = [FLAGS.batch_size, image_size, image_size, 3]
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, shape=batch_shape)
        model = model_selector(FLAGS.model_name)
        x_adv = adv_attack(model, inputs, eps=FLAGS.eps, clip_min=FLAGS.clip_min, clip_max=FLAGS.clip_max)
        saver = tf.train.Saver(slim.get_model_variables())
        if FLAGS.model_name == "inceptionv3":
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=tf.train.Scaffold(saver=saver),
                checkpoint_filename_with_path=FLAGS.checkpoint_path, master=FLAGS.master)
        else:
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=tf.train.Scaffold(saver=saver),
                checkpoint_dir=FLAGS.checkpoint_path, master=FLAGS.master)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for root, dirs, _ in os.walk(FLAGS.input_dir):
                for d in dirs:
                    for filenames, images in load_images(os.path.join(root, d), batch_shape):
                        adv_images = sess.run(x_adv, feed_dict={inputs: images})
                        if not os.path.exists(FLAGS.output_dir):
                            os.mkdir(FLAGS.output_dir)
                        if not os.path.exists(os.path.join(FLAGS.output_dir, d)):
                            os.mkdir(os.path.join(FLAGS.output_dir, d))
                        save_images(adv_images, filenames, os.path.join(FLAGS.output_dir, d))


if __name__ == '__main__':
    tf.app.run()

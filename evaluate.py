import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import MobileNet as mn
from utils import get_flags, read_single_image, get_image_size

FLAGS = get_flags(method="evaluate")


def main(_):
    image_size = get_image_size(FLAGS.model_name)
    batch_shape = [2, image_size, image_size, 3]
    tf.logging.set_verbosity(tf.logging.INFO)
    ori_dir = FLAGS.input_dir
    adv_dir = FLAGS.adv_example_dir
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        inputs = tf.placeholder(tf.float32, shape=batch_shape)
        if FLAGS.model_name == "mobilenet":
            with slim.arg_scope(mn.mobilenet_v1_arg_scope()):
                logits, endpoints = mn.mobilenet_v1(inputs=inputs, is_training=False, num_classes=1000,
                                                    spatial_squeeze=True, reuse=None)
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_dir=FLAGS.checkpoint_path)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filename in os.listdir(adv_dir):
                test_batch = np.zeros(batch_shape)
                test_batch[0, :, :, :] = read_single_image(os.path.join(ori_dir, filename), image_size)
                test_batch[1, :, :, :] = read_single_image(os.path.join(adv_dir, filename), image_size)
                output = sess.run(endpoints, feed_dict={inputs: test_batch})
                print(np.argmax(output["Logits"], 1))



if __name__ == "__main__":
    tf.app.run()

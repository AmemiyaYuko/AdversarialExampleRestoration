import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import softmax

import MobileNet as mn
from utils import get_flags, read_single_image

FLAGS = get_flags(method="evaluate")


def main(_):
    batch_shape = [2, FLAGS.image_size, FLAGS.image_size, 3]
    tf.logging.set_verbosity(tf.logging.INFO)
    print(FLAGS.checkpoint_path)
    ori_dir = FLAGS.input_dir
    adv_dir = FLAGS.adv_example_dir
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, shape=batch_shape)
        with slim.arg_scope(mn.mobilenet_v1_arg_scope()):
            logits, endpoints = mn.mobilenet_v1(inputs=inputs, is_training=False, num_classes=1000,
                                                spatial_squeeze=True, reuse=None, prediction_fn=softmax)
            prediction = endpoints["Predictions"]
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_dir=FLAGS.checkpoint_path)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filename in os.listdir(adv_dir):
                test_batch = np.zeros(batch_shape)
                test_batch[0, :, :, :] = read_single_image(os.path.join(ori_dir, filename), FLAGS.image_size)
                test_batch[1, :, :, :] = read_single_image(os.path.join(adv_dir, filename), FLAGS.image_size)
                output = sess.run(endpoints, feed_dict={inputs: test_batch})
                for line in output["Logits"]:
                    print(np.argmax(line))
                print("asd")


if __name__ == "__main__":
    tf.app.run()

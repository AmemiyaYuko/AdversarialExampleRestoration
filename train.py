import os

import cv2
import tensorflow as tf

from restore_model import model as res_model
from utils import get_flags, get_image_size

FLAGS = get_flags("train")


def _preprocess_dataset(ori, adv):
    image_size = get_image_size(FLAGS.model_name)
    ori_str = tf.read_file(ori)
    adv_str = tf.read_file(adv)
    ori_decoded = tf.image.decode_jpeg(ori_str, channels=3)
    adv_decoded = tf.image.decode_jpeg(adv_str, channels=3)
    ori_resized = tf.image.resize_images(ori_decoded, [image_size, image_size])
    adv_resized = tf.image.resize_images(adv_decoded, [image_size, image_size])
    return ori_resized / 255 * 2 - 1, adv_resized / 255 * 2 - 1


def get_database(adv_path, ori_path):
    ori_files = []
    adv_files = []
    for root, dirs, filenames in os.walk(ori_path):
        for f in os.listdir(os.path.join(root)):
            ori_image = os.path.join(root, f)
            adv_image = os.path.join(adv_path, f)
            ori_files.append(ori_image)
            adv_files.append(adv_image)
    ori_examples = tf.constant(ori_files)
    adv_examples = tf.constant(adv_files)
    dataset = tf.data.Dataset.from_tensor_slices((ori_examples, adv_examples))
    dataset = dataset.map(_preprocess_dataset)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.shuffle(buffer_size=FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def main(_):
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(r"E:\Dropbox\Dropbox\Code\AdversarialExampleRestoration\logger", sess.graph)
        iterator = get_database(r"D:\2010_adv", r"D:\ILSVRC2012_img_val")
        ori, adv = iterator.get_next()
        model = res_model(ori, adv, is_trainging=True)
        sess.run(tf.global_variables_initializer())
        for i in range(1000000):
            _, summary, step, loss, images = sess.run([model["optimizer"], model["summary"],
                                                       model["global_step"], model["loss"], model["output"]])
            writer.add_summary(summary, step)
            print(loss)
            writer.flush()
            if (i % 100 == 1):
                cv2.imwrite("%d_1.jpg" % i, postprocess(images[0]))
                cv2.imwrite("%d_2.jpg" % i, postprocess(images[1]))
                cv2.imwrite("%d_3.jpg" % i, postprocess(images[2]))
                cv2.imwrite("%d_4.jpg" % i, postprocess(images[3]))


def postprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return (img + 1) * 255
if __name__ == "__main__":
    tf.app.run()

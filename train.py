import os

import tensorflow as tf

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
        for d in dirs:
            for f in os.listdir(os.path.join(root, d)):
                ori_image = os.path.join(root, d, f)
                adv_image = os.path.join(adv_path, d, f)
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
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    iterator = get_database(r"/home/qide/imagenet-data/raw-data/validation", r"/home/qide/output")
    ori, adv = sess.run(iterator.get_next())

if __name__ == "__main__":
    tf.app.run()

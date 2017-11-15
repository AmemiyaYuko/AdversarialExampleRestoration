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
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def main(_):
    with tf.train.MonitoredSession() as sess:
        saver = tf.train.Saver()
        iterator = get_database(r"D:\2010_adv", r"D:\ILSVRC2012_img_val")
        ori, adv = iterator.get_next()
        model = res_model(ori, adv, is_trainging=False)
        sess.run(tf.global_variables_initializer())
        ckpt_path = r"D:\checkpoints\step019501"
        ckpt = tf.train.latest_checkpoint(checkpoint_dir=ckpt_path)
        saver.restore(sess, ckpt)
        counter = 0
        total_psnr = 0.
        total_mse = 0.
        while not sess.should_stop():
            images, residual, mse, psnr, ori_image, adv_image = sess.run(
                [model["output"], model["residual"], model["mse"], model['psnr'], model["ori_image"],
                 model["adv_image"]])
            total_psnr += psnr
            total_mse += mse
            for j in range(len(images)):
                counter = counter + 1
                cv2.imwrite(os.path.join(dir, "%06d_full.jpg" % counter), postprocess(images[j]))
                cv2.imwrite(os.path.join(dir, "%06d_res.jpg" % j), postprocess(residual[j]))
                cv2.imwrite(os.path.join(dir, "%06d_ori.jpg" % j), postprocess(ori_image[j]))
                cv2.imwrite(os.path.join(dir, "%06d_adv.jpg" % j), postprocess(adv_image[j]))
        print("%d images", counter)


def postprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return (img + 1.0) * 255.0 / 2.0


if __name__ == "__main__":
    tf.app.run()

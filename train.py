import os
import shutil
from time import time

import cv2
import tensorflow as tf

from restore_model import model as res_model
from utils import get_flags, get_image_size

FLAGS = get_flags("train")


def _preprocess_dataset(ori, adv, file_names):
    image_size = get_image_size(FLAGS.model_name)
    ori_str = tf.read_file(ori)
    adv_str = tf.read_file(adv)
    ori_decoded = tf.image.decode_jpeg(ori_str, channels=3)
    adv_decoded = tf.image.decode_jpeg(adv_str, channels=3)
    ori_resized = tf.image.resize_images(ori_decoded, [image_size, image_size])
    adv_resized = tf.image.resize_images(adv_decoded, [image_size, image_size])
    return ori_resized / 255 * 2 - 1, adv_resized / 255 * 2 - 1, file_names


def get_database(adv_path, ori_path):
    ori_files = []
    adv_files = []
    file_names = []
    for root, dirs, filenames in os.walk(ori_path):
        for f in os.listdir(os.path.join(root)):
            ori_image = os.path.join(root, f)
            adv_image = os.path.join(adv_path, f)
            ori_files.append(ori_image)
            adv_files.append(adv_image)
            file_names.append(f)
    ori_examples = tf.constant(ori_files)
    adv_examples = tf.constant(adv_files)
    filenames_tensor = tf.constant(file_names, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((ori_examples, adv_examples, filenames_tensor))
    dataset = dataset.map(_preprocess_dataset)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.shuffle(buffer_size=FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def main(_):
    with tf.Session() as sess:
        epoch = 0
        logdir = r"logger"
        if not os.path.isdir(logdir):
            shutil.rmtree(logdir)
            os.mkdir(logdir)

        iterator = get_database(r"D:\2010_adv", r"D:\ILSVRC2012_img_val")

        model = res_model(is_trainging=FLAGS.is_training)
        writer = tf.summary.FileWriter(logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i in range(30002):
            ori, adv, file_name = sess.run(iterator.get_next())
            start = time()
            _, summary, step, loss, images, residual, mse, psnr = sess.run([model["optimizer"], model["summary"],
                                                                            model["global_step"], model["loss"],
                                                                            model["output"],
                                                                            model["residual"], model["mse"],
                                                                            model['psnr']],
                                                                           feed_dict={model["ori_image"]: ori,
                                                                                      model["adv_image"]: adv})
            fid = file_name

            start = time() - start
            writer.add_summary(summary, step)
            writer.flush()
            print(
                "step %5d with loss %.8f takes time %.3f seconds. MSE= %.3f, PSNR= %.3f" % (i, loss, start, mse, psnr))
            if (i % 2000 == 1):
                save_path = os.path.join(FLAGS.checkpoint_path, "step%06d" % i, "")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                result = saver.save(sess, os.path.join(save_path, "model.ckpt"))
                print("saved on " + result)
                dir = r"D:\new_output\\" + str("%06d" % i)
                if not os.path.exists(dir):
                    os.mkdir(dir)
                for j in range(len(images)):
                    name = str(fid[j]).split(".")[0].split("'")[-1]
                    cv2.imwrite(os.path.join(dir, "full_%s.jpg" % name), postprocess(images[j]))
                    cv2.imwrite(os.path.join(dir, "res_%s.jpg" % name), postprocess(residual[j]))
            if (i % 50000 == 1):
                f = open("epoch_record.txt", "a+")
                f.write("Epoch:%s,MSE:%.3f,PSNR=%.3f\n" % (epoch, mse, psnr))
                epoch += 1
                f.close()
                print("epoch record updated")
    sess.close()


def postprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return (img + 1.0) * 255.0 / 2.0


if __name__ == "__main__":
    tf.app.run()

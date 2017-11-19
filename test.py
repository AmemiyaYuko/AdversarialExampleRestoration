import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from restore_model import model as res_model
from utils import get_flags

FLAGS = get_flags("train")
_batch_size = 20


def _preprocess_dataset(ori):
    with tf.gfile.Open(ori, mode="rb") as f:
        image = Image.open(f).convert('RGB')
        image = Image.Image.resize(image, (299, 299))
        image = np.array(image).astype(np.float)

    return image / 255.0 * 2.0 - 1.0

def get_database(adv_path, ori_path):
    ori_files = np.zeros([_batch_size, 299, 299, 3])
    adv_files = np.zeros([_batch_size, 299, 299, 3])
    file_names = []
    idx = 0
    for f in os.listdir(adv_path):
        ori_image = _preprocess_dataset(os.path.join(ori_path, f))
        adv_image = _preprocess_dataset(os.path.join(adv_path, f))
        ori_files[idx, :, :, :] = ori_image
        adv_files[idx, :, :, :] = adv_image
        file_names.append(f)
        idx += 1
        if idx == _batch_size:
            yield (ori_files, adv_files, file_names)
            ori_files = np.zeros([_batch_size, 299, 299, 3])
            adv_files = np.zeros([_batch_size, 299, 299, 3])
            file_names = []
            idx = 0
    if (idx > 0):
        yield (ori_files, adv_files, file_names)




def main(_):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ori = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
            adv = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
            model = res_model(ori, adv, None, is_trainging=False)
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            ckpt_path = r"./ckpt/"
            saver = tf.train.Saver()
            state = tf.train.get_checkpoint_state(tf.train.latest_checkpoint(ckpt_path))
            print(state)
            print(saver.restore(sess=sess, save_path=state.model_checkpoint_path))
            for ori_data, adv_data, files in get_database(r"D:\2010_adv", r"D:\ILSVRC2012_img_val"):
                images = sess.run(
                    model["output"], feed_dict={ori: ori_data, adv: adv_data})
                for j in range(len(images)):
                    dir = r"D:\test_output_nosc"
                    name = str(files[j]).split(".")[0].split("'")[-1]
                    cv2.imwrite(os.path.join(dir, "%s_full.png" % name), postprocess(images[j]))
                    save_image(os.path.join(dir, "%s_ori.jpg" % name), ori_data[j])
                    save_image(os.path.join(dir, "%s_adv.jpg" % name), adv_data[j])
                    # cv2.imwrite(os.path.join(dir, "%s_ori.jpg" % name), postprocess(ori_data[j]).astype(np.uint8))
                    # cv2.imwrite(os.path.join(dir, "%s_adv.jpg" % name), postprocess(adv_data[j]).astype(np.uint8))

def postprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return (img + 1.0) * 255.0 / 2.0


def save_image(file_name, image):
    with tf.gfile.Open(file_name, 'wb') as f:
        img = (((image[:, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        Image.fromarray(img).save(f, format='PNG')


if __name__ == "__main__":
    tf.app.run()

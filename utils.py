import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
from tensorflow.contrib.slim.nets import inception


def load_images(input_dir, batch_shape):
    ''' input images generator '''
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, mode="rb") as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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
        with tf.gfile.Open(os.path.join(output_dir, filename), 'wb') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def get_category(label,
                 csv_path=r'C:\Datasets\Kaggle\nips2017_adversarial_attack\nips-2017-adversarial-learning-development-set\categories.csv'):
    df_cate = pd.DataFrame(pd.read_csv(csv_path))
    return str(df_cate[df_cate['CategoryId'] == label])


def predict_images(img_paths,
                   ckpt_path='C:/Datasets/Kaggle/nips2017_adversarial_attack/inception-v3/inception_v3.ckpt'):
    i = 0
    imgs = np.zeros([len(img_paths), 299, 299, 3])
    for img_path in img_paths:
        img_file = open(img_path, "rb")
        img = np.array(Image.open(img_file).convert('RGB')).astype(np.float) / 255.0
        imgs[i, :, :, :] = img * 2.0 - 1.0
        i += 1
    x_input = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points = inception.inception_v3(
            x_input, num_classes=1001, is_training=False,
            reuse=None)
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=ckpt_path)
    output = []
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        for i in range(20):
            output_ori = sess.run(end_points, feed_dict={x_input: imgs[i * 100:i * 100 + 99]})
            for j in range(100):
                output.append(np.argmax(output_ori['Predictions']))
    return output

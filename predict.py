import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
from tensorflow.contrib.slim.nets import inception

# img_path = r'C:\Datasets\Kaggle\nips2017_adversarial_attack\nips-2017-adversarial-learning-development-set\images\1a95932427129cda.png'
pic_name = '70b3b833463d20b9.png'
adv_img_path = r'e:\output\\' + pic_name
ori_img_path = r'C:\Datasets\Kaggle\nips2017_adversarial_attack\nips-2017-adversarial-learning-development-set\images\\' + pic_name
checkpoint_path = 'C:/Datasets/Kaggle/nips2017_adversarial_attack/inception-v3/inception_v3.ckpt'
imagelabel_path = r'C:\Datasets\Kaggle\nips2017_adversarial_attack\nips-2017-adversarial-learning-development-set\images.csv'
categories_path = r'C:\Datasets\Kaggle\nips2017_adversarial_attack\nips-2017-adversarial-learning-development-set\categories.csv'
adv = cv2.imread(adv_img_path)
adv = cv2.bilateralFilter(adv, 9, 75, 75)
cv2.imwrite("test.jpg", adv)
adv_img_path = "test.jpg"
cv2.imshow("test", adv)
cv2.waitKey(0)
ori_img_file = open(ori_img_path, "rb")
adv_img_file = open(adv_img_path, "rb")
ori_img = np.array(Image.open(ori_img_file).convert('RGB')).astype(np.float) / 255.0
adv_img = np.array(Image.open(adv_img_file).convert('RGB')).astype(np.float) / 255.0
ori_images = np.zeros([1, 299, 299, 3])
ori_images[0, :, :, :] = ori_img * 2.0 - 1.0
adv_images = np.zeros([1, 299, 299, 3])
adv_images[0, :, :, :] = adv_img * 2.0 - 1.0
x_input = tf.placeholder(tf.float32, shape=[1, ori_img.shape[0], ori_img.shape[1], ori_img.shape[2]])
with slim.arg_scope(inception.inception_v3_arg_scope()):
    _, end_points = inception.inception_v3(
        x_input, num_classes=1001, is_training=False,
        reuse=None)
saver = tf.train.Saver(slim.get_model_variables())
session_creator = tf.train.ChiefSessionCreator(
    scaffold=tf.train.Scaffold(saver=saver),
    checkpoint_filename_with_path=checkpoint_path)
with tf.train.MonitoredSession(session_creator=session_creator) as sess:
    ori_output = sess.run(end_points, feed_dict={x_input: ori_images})
    adv_output = sess.run(end_points, feed_dict={x_input: adv_images})
ori_label = np.argmax(ori_output['Predictions'][0])
adv_label = np.argmax(adv_output['Predictions'][0])
df_cate = pd.DataFrame(pd.read_csv(categories_path))
output_text = "left: \n" + str(df_cate[df_cate['CategoryId'] == ori_label]) + "\nright: \n" + str(
    df_cate[df_cate['CategoryId'] == adv_label])
plt.figure()
plt.title("Compare adversarial example to original image")
plt.imshow(np.concatenate((Image.open(ori_img_file).convert('RGB'), Image.open(adv_img_file).convert('RGB')), axis=1))
plt.axis('off')
print(output_text)
plt.show()

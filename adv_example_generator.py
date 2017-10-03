import tensorflow as tf
from cleverhans.attacks import FastGradientMethod

tf.flags.DEFINE_integer("batch_size", 32, 'How many images to be processed at one time')
tf.flags.DEFINE_integer("image_size", 32, 'Size of images')
FLAG = tf.flags.FLAGS


def fgsm_attack(prediction, eps=0.2, clip_min=-1.0, clip_max=1.0):
    images = tf.placeholder(tf.float32, shape=[FLAG.batch_size, FLAG.image_size, FLAG.image_size, 3])
    fgsm = FastGradientMethod(prediction)
    adv = fgsm.generate(images, eps=eps, clip_min=clip_min, clip_max=clip_max)
    return adv

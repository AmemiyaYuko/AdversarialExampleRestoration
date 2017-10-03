import os
import shutil

import numpy as np
import tensorflow as tf

from cifar10 import load_cifar10
from model_without_slim import model


def preprocessing():
    train_images, train_labels, test_images, test_labels = load_cifar10(
        prefix=r"/home/qide/Dataset/cifar-10-batches-py")
    train_images /= 255.0
    train_images = train_images * 2 - 1
    test_images /= 255.0
    test_images = test_images * 2 - 1
    train_labels = (np.arange(10) == train_labels[:, None]).astype(np.float32)
    test_labels = (np.arange(10) == test_labels[:, None]).astype(np.float32)
    return train_images, train_labels, test_images, test_labels


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def train(batch_size, max_epoch, perform_clean=False):
    if perform_clean:
        if os.path.exists(os.path.join(os.getcwd(), "logger")):
            shutil.rmtree(os.path.join(os.getcwd(), "logger"))
        if os.path.exists(os.path.join(os.getcwd(), "ckpt")):
            shutil.rmtree(os.path.join(os.getcwd(), "ckpt"))
    train_images, train_labels, test_images, test_labels = preprocessing()
    with tf.Session() as sess:
        graph = model()
        saver = tf.train.Saver()
        trn_summary = tf.summary.FileWriter('logger/train/', sess.graph)
        tst_summary = tf.summary.FileWriter('logger/test/')
        sess.run(tf.global_variables_initializer())
        for i in range(max_epoch):
            if (i % 10 == 0):
                saver.save(sess, "ckpt/", global_step=graph['global_step'])
            gstep = 0
            shuffled_images, shuffled_labels = unison_shuffled_copies(train_images, train_labels)
            # train_images, train_labels=unison_shuffled_copies(train_images,train_labels)
            for j in range(len(train_images) // batch_size):
                trn_imgs = shuffled_images[j * batch_size:(j + 1) * batch_size]
                trn_labels = shuffled_labels[j * batch_size:(j + 1) * batch_size]
                graph_dict = {graph["x"]: trn_imgs,
                              graph["y"]: trn_labels,
                              graph["keep_prob"]: 0.5,
                              graph['is_training']: True}
                opt, loss, summary, step = sess.run(
                    [graph['optimizer'], graph['loss'], graph['summary'], graph['global_step']], feed_dict=graph_dict)
                trn_summary.add_summary(summary, step)
                gstep = step
                print("# Epoch " + str(i) + " Step " + str(step) + " with loss " + str(loss))

            # test

            graph_dict = {graph["x"]: test_images[:1000],
                          graph["y"]: test_labels[:1000],
                          graph["keep_prob"]: 1,
                          graph["is_training"]: False
                          }
            accuracy, summary = sess.run([graph['accuracy'], graph['summary']], feed_dict=graph_dict)
            tst_summary.add_summary(summary, global_step=gstep)
            print("Accuracy in epoch #" + str(i) + " = " + str(accuracy))
            print('------------------\n')

    sess.close()


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def get_weights(name):
    # in order to get weights, sess.run the return value of this function
    all_vars = tf.trainable_variables()
    for i in range(len(all_vars)):
        if all_vars[i].name.startswith(name):
            return all_vars[i]
    return None


if __name__ == "__main__":
    train(256, 100, perform_clean=True)

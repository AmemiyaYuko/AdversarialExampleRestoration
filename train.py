import numpy as np
import tensorflow as tf

from cifar10 import load_cifar10
from model import model


def preprocessing():
    train_images, train_labels, test_images, test_labels = load_cifar10(
        prefix=r"/home/qide/Dataset/cifar-10-batches-py/")
    train_images /= 255.0
    test_images /= 255.0
    train_labels = (np.arange(10) == train_labels[:, None]).astype(np.float32)
    test_labels = (np.arange(10) == test_labels[:, None]).astype(np.float32)
    return train_images, train_labels, test_images, test_labels


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def train(batch_size, max_epoch):
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
            # train_images, train_labels=unison_shuffled_copies(train_images,train_labels)
            for j in range(len(train_images) // batch_size):
                trn_imgs = train_images[j * batch_size:(j + 1) * batch_size]
                trn_labels = train_labels[j * batch_size:(j + 1) * batch_size]
                graph_dict = {graph["x"]: trn_imgs,
                              graph["y"]: trn_labels,
                              graph["keep_prob"]: 0.5}
                opt, loss, summary, step = sess.run(
                    [graph['optimizer'], graph['loss'], graph['summary'], graph['global_step']], feed_dict=graph_dict)
                trn_summary.add_summary(summary, step)
                gstep = step
                print("# Epoch " + str(i) + " Step " + str(step) + " with loss " + str(loss))

            # test
            graph_dict = {graph["x"]: test_images[:1000],
                          graph["y"]: test_labels[:1000],
                          graph["keep_prob"]: 1}
            accuracy, summary = sess.run([graph['accuracy'], graph['summary']], feed_dict=graph_dict)
            tst_summary.add_summary(summary, global_step=gstep)
            print("Accuracy in epoch #" + str(i) + " = " + str(accuracy))

    sess.close()


if __name__ == "__main__":
    train(512, 1000)

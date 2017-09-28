import os

import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_all_images(files):
    all_imgs = []
    all_labels = []
    for file_name in files:
        d = unpickle(file_name)
        images = d[b"data"]
        for i in range(images.shape[0]):
            img = images[i]
            channel_r = img[:32 * 32]
            channel_g = img[32 * 32:2 * 32 * 32]
            channel_b = img[2 * 32 * 32:]
            output_img = np.zeros([32, 32, 3], dtype=np.float32)
            output_img[:, :, 0] = channel_r.reshape([32, 32])
            output_img[:, :, 1] = channel_g.reshape([32, 32])
            output_img[:, :, 2] = channel_b.reshape([32, 32])
            all_imgs.append(output_img)
            all_labels.append(d[b"labels"][i])
    return np.asarray(all_imgs, dtype=np.float32), np.asarray(all_labels, dtype=np.float32)


def load_cifar10(prefix):
    train_images, train_labels = load_all_images(
        [os.path.join(prefix, r"data_batch_1"), os.path.join(prefix, r"data_batch_2"),
         os.path.join(prefix, r"data_batch_3"), os.path.join(prefix, r"data_batch_4"),
         os.path.join(prefix, r"data_batch_5")])
    test_images, test_labels = load_all_images([os.path.join(prefix, r"test_batch")])
    return train_images, train_labels, test_images, test_labels

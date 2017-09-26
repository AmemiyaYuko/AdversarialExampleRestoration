import cv2
import numpy as np


def unpickle(file=r"C:\Datasets\cifar10\data_batch_1"):
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
            output_img = np.zeros([32, 32, 3], dtype=np.uint8)
            output_img[:, :, 0] = channel_r.reshape([32, 32])
            output_img[:, :, 1] = channel_g.reshape([32, 32])
            output_img[:, :, 2] = channel_b.reshape([32, 32])
            all_imgs.append(output_img)
            all_labels.append(d[b"label"][i])
            cv2.imshow("test", output_img)
            print(d[b"label"][i])
            cv2.waitKey()

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from read_mat_file import ground_truth


class FaceDetectionDataloader(Sequence):
    def __init__(self, img_data_path, batch_size: int, image_size: tuple):
        # self.images = []
        # self.labels = []
        self.labels = ground_truth()
        self.filenames = list(self.labels.keys())
        self.path = img_data_path
        self.batch_size = batch_size

        self.width, self.height = image_size

        # gt = ground_truth()
        # filenames = list(gt.keys())
        # for filename in tqdm(filenames, f"[Data Load]:"):
        #     label = gt[filename]
        #     img = tf.io.read_file(f"./{img_data_path}/{filename}")
        #     img = tf.io.decode_image(img)
        #
        #     self.images.append(img)
        #     self.labels.append(label)

    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):
        indices = list(range(idx * self.batch_size, (idx+1) * self.batch_size))
        # batch_img = [self.images[i] for i in indices]
        # batch_label = [self.labels[i] for i in indices]

        img = [tf.io.decode_image(
            tf.io.read_file(f"{self.path}/{self.filenames[i]}")
        ) for i in indices]

        # 강제 resize
        batch_img = [tf.image.resize(
            img[i],
            [self.width, self.height]
        ) for i in indices]
        batch_label = [self.resize_label(
            self.labels[self.filenames[i]], img[i].shape
        ) for i in indices]
        
        filenames = [self.filenames[i] for i in indices]

        return np.array(batch_img), np.array(batch_label), filenames

    def resize_label(self, labels, origin_shape):
        # origin_w x origin_h -> 96x96 resize에 맞춰 좌표 변경
        # label, origin_shape의 shape -> (4,)

        # 이거처럼 하는게 맞는거같은데 gt bounding box의 x y 좌표 위치가 이상함
        # new_label = [
        #     [
        #         label[0] / origin_shape[0] * self.width,  # x
        #         label[1] / origin_shape[1] * self.height,  # y
        #         label[2] / origin_shape[0] * self.width,  # width
        #         label[3] / origin_shape[1] * self.height,  # height
        #     ] for label in labels
        # ]
        new_label = [
            [
                label[0] / origin_shape[1] * self.width,  # x
                label[1] / origin_shape[0] * self.height,  # y
                label[2] / origin_shape[1] * self.width,  # width
                label[3] / origin_shape[0] * self.height,  # height
            ] for label in labels
        ]
        return new_label


class StaticTilingDataloader(Sequence):
    def __init__(self, img_data_path, batch_size: int,
                 image_size: tuple, tile_size: tuple):
        self.labels = ground_truth()
        self.filenames = list(self.labels.keys())
        self.path = img_data_path
        self.batch_size = batch_size

        self.n, self.m = tile_size
        self.width, self.height = image_size

    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):
        indices = list(range(idx * self.batch_size, (idx + 1) * self.batch_size))
        batch_label = [self.labels[i] for i in indices]

        imgs = [tf.io.decode_image(
            tf.io.read_file(f"{self.path}/{self.filenames[i]}")
        ) for i in indices]
        croped_imgs = [self.crop_image(img) for img in imgs]
        filenames = [self.filenames[i] for i in indices]

        return np.array(croped_imgs), np.array(batch_label), filenames

    def idx2box(self, w_idx, h_idx, image_shape):
        # image_shape = (height x width) <- 벡터의 shape를 보내므로
        x = math.ceil(w_idx * image_shape[1] / self.n)
        y = math.ceil(h_idx * image_shape[0] / self.m)
        w = math.ceil((w_idx + 1) * image_shape[1] / self.n)
        m = math.ceil((h_idx + 1) * image_shape[0] / self.m)
        return x, y, w, m

    def crop_image(self, img):
        imgs = [
            img[self.idx2box(_w, _h, img.shape)[0]:
                self.idx2box(_w, _h, img.shape)[2],
                self.idx2box(_w, _h, img.shape)[1]:
                self.idx2box(_w, _h, img.shape)[3]]
            for _w in range(self.n) for _h in range(self.m)
        ]
        return imgs

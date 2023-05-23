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
        with open("data/over_5_face.txt", "r") as fp:
            self.filenames = fp.read()
            self.filenames = self.filenames.split("\n")[:-1]
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

        if "test" in self.path:
            img = [tf.io.decode_image(
                tf.io.read_file(f"{self.path}/0_Parade_marchingband_1_476.jpg")
            ) for i in indices]
        else:
            img = [tf.io.decode_image(
                tf.io.read_file(f"{self.path}/{self.filenames[i]}")
            ) for i in indices]

        origin_size = [
            img[i].get_shape().as_list() for i in range(len(img))
        ]
        # 강제 resize
        batch_img = [tf.image.resize(
            img[i] / 255,       # normalize
            [self.width, self.height]
        ) for i in range(len(img))]
        # batch_label = [self.resize_label(
        #     self.labels[self.filenames[i]], img[_i].shape
        # ) for i, _i in zip(indices, range(len(img)))]
        batch_label = [
            self.labels[self.filenames[i]] for i in indices
        ]

        filenames = [self.filenames[i] for i in indices]

        # return np.array(batch_img), np.array(batch_label), filenames, img
        return np.array(batch_img), np.array(batch_label), filenames, None, origin_size

    def resize_label(self, labels, origin_shape):
        # origin_w x origin_h -> 96x96 resize에 맞춰 좌표 변경
        # label, origin_shape의 shape -> (4,)
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

        # 데이터 개수 제한
        with open("data/under_5_face.txt", "r") as fp:
            self.filenames = fp.read()
            self.filenames = self.filenames.split("\n")[:-1]
        self.path = img_data_path
        self.batch_size = batch_size

        self.n, self.m = tile_size
        self.width, self.height = image_size
        self.tile_size = list(tile_size)

    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):
        indices = list(range(idx * self.batch_size, (idx + 1) * self.batch_size))
        batch_label = [self.labels[self.filenames[i]] for i in indices]
        imgs = [tf.io.decode_image(
            tf.io.read_file(f"{self.path}/{self.filenames[i]}")
        ) / 255 for i in indices]

        origin_size = [
            img.get_shape().as_list() for img in imgs
        ]
        cropped_imgs = [self.crop_image(img) for img in imgs]
        filenames = [self.filenames[i] for i in indices]

        return np.array(cropped_imgs), np.array(batch_label), filenames, self.tile_size, origin_size

    def crop_image(self, img):
        # img.shape = col x row
        col, row, ch = img.shape
        imgs = [
            # img[math.ceil(row / self.n) * _w:math.ceil(row / self.n) * (_w + 1),
            #     math.ceil(col / self.m) * _h:math.ceil(col / self.m) * (_h + 1)]
            img[math.ceil(col / self.m) * _h:math.ceil(col / self.m) * (_h + 1),
                math.ceil(row / self.n) * _w:math.ceil(row / self.n) * (_w + 1)]
            for _w in range(self.n) for _h in range(self.m)
        ]
        imgs = [
            tf.image.resize(img, [96, 96])
            for img in imgs
        ]

        return imgs

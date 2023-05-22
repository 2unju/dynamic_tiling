from PIL import Image, ImageDraw
import os
import math
import tensorflow as tf
import numpy as np

from utils import *


def draw_n_save(img, bbx, idx):
    # img = Image.open(f"data/valid/{filename}")
    # img_vector = tf.io.decode_image(
    #        tf.io.read_file(f"data/valid/{filename}")
    # )
    # print(img_vector.shape)
    img[0] *= 255
    img = Image.fromarray(img[0], "RGB")

    draw = ImageDraw.Draw(img)
    for box in bbx:
        if type(box) == tuple:
            box = list(box)
        box[2] += box[0]
        box[3] += box[1]
        _box = box
        box = [_box[0], _box[1], _box[2], _box[3]]
        draw.rectangle(box, outline=(255, 0, 0))
        # print(box)
        # break
    os.makedirs("result/draw/test/", exist_ok=True)
    img.save(f"result/draw/test/cropped_{idx}.jpg")


model = tf.lite.Interpreter(model_path="./tflite/fomo_face_detection.tflite")
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

input_shape = input_details[0]["shape"]
output_shape = output_details[0]["shape"]
input_dtype = input_details[0]["dtype"]
input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

width = input_shape[1]
height = input_shape[2]
output_width = output_shape[1]
output_height = output_shape[2]

batch_size = 1
iou_threshold = 0.1
tile_size = (2, 2)

filename = "2_Demonstration_Demonstration_Or_Protest_2_225.jpg"
img = tf.io.decode_image(
       tf.io.read_file(f"data/valid/{filename}")
)
# print(img.shape)
# _img = Image.fromarray(img.numpy(), "RGB")
# _img.save(f"result/draw/test/read_image.jpg")
# exit()

# 1. common
# img = tf.io.decode_image(
#        tf.io.read_file(f"data/valid/{filename}")
# ) / 255
#
# img = tf.image.resize(img, [96, 96])
#
# img = real2quantized(img, input_zero_point, input_scale)
# img = np.expand_dims(img, axis=0).astype(input_dtype)
#
# model.set_tensor(input_details[0]["index"], img)
# model.invoke()
#
# output_data = model.get_tensor(output_details[0]["index"])
# output_data = quantized2real(output_data, output_zero_point, output_scale)
#
# print(output_data)

# 2. tiling
img = tf.io.decode_image(
       tf.io.read_file(f"data/valid/{filename}")
) / 255

col, row, ch = img.shape
n, m = 2, 2
imgs = [
       img[math.ceil(col / m) * _h:math.ceil(col / m) * (_h + 1),
       math.ceil(row / n) * _w:math.ceil(row / n) * (_w + 1)]
       for _w in range(n) for _h in range(m)
]
imgs = [
       tf.image.resize(img, [96, 96])
       for img in imgs
]

pred_bbs = []
coordinate = []

for idx, img in enumerate(imgs):
    img = real2quantized(img, input_zero_point, input_scale)
    img = np.expand_dims(img, axis=0).astype(input_dtype)

    model.set_tensor(input_details[0]["index"], img)
    model.invoke()

    output_data = model.get_tensor(output_details[0]["index"])
    output_data = quantized2real(output_data, output_zero_point, output_scale)

    print(output_data)
    pred_bbs = []

    for h_idx, preds in enumerate(output_data[0]):
        for w_idx, pred in enumerate(preds):
            if pred[1] > 0.1:
                # print(f"w_idx, h_idx: {w_idx, h_idx}")
                coordinate.append([w_idx, h_idx])
                pred_bbs.append(
                    idx2boundingbox(w_idx, h_idx, width, height, output_width, output_height))
    draw_n_save(img, pred_bbs, idx)

print(coordinate)
# draw_n_save(filename, pred_bbs)
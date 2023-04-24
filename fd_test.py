import os
from PIL import Image
import tensorflow as tf
import numpy as np

# from read_mat_file import ground_truth
from dataloader import FaceDetectionDataloader


def real2quantized(real_value, zero_point, scale):
    return (real_value / scale) + zero_point


def quantized2real(quantized_value, zero_point, scale):
    return (quantized_value - zero_point) * scale


# custon dataset -> 2번의 non-rectangular Python sequence 에러 해결 실패
# 왠지 filenames와 labels의 차원이 맞춰져서
# <BatchDataset element_spec=(TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 1, 4), dtype=tf.int32, name=None))>
# file을 읽어오지 못함
# def _parse_function(filename, label):
#     img_string = tf.io.read_file(f"data/valid/{filename}")
#     img_decoded = tf.image.decode_jpeg(img_string, channels=3)
#     img = tf.cast(img_decoded, tf.float32)
#     return img, label
#
#
# def read_image_dataset():
#     # code from https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
#     gt = ground_truth()
#
#     filenames = list(gt.keys())
#     labels = [gt[key] for key in filenames]
#
#     dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.batch(2)
#     return dataset
#
# dataset = read_image_dataset()
# print(dataset)
# for elem in dataset:
#     print(elem)
#     exit()


model = tf.lite.Interpreter(model_path="./tflite/fomo_face_detection.tflite")
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

print(input_details, end="\n\n")
print(output_details, end="\n\n")

input_shape = input_details[0]["shape"]
output_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]
input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

width = input_shape[1]
height = input_shape[2]
batch_size = 1

# gt = ground_truth()
#
# tf 내부 함수를 이용하여 데이터 로드
# '4월 3주차 issue.txt 내 2번 문제 해결을 못함 -> custom dataloader 생성
# valid_data = tf.keras.utils.image_dataset_from_directory(
#     "./data/WIDER_val/images",
#     labels="inferred",
#     batch_size=batch_size,
#     image_size=(width, height),
#     shuffle=False
# )
#
# # read ground truth
#
# for img, labels in valid_data.take(1):
#     print(img)
#     # print(labels)
#     exit()
#     _img = real2quantized(img, input_zero_point, input_scale)
#     # _img = np.expand_dims(_img, axis=0).astype(input_dtype)
#     _img = tf.cast(_img, input_dtype)
#     print(_img)
#
#     model.set_tensor(input_details[0]["index"], _img)
#     model.invoke()
#
#     output_data = model.get_tensor(output_details[0]["index"])
#     # output_data = quantized2real(output_data, output_zero_point, output_scale)
#     print(output_data)
#     print(np.array(output_data).shape)
#     # print(len(output_data))
#     exit()

dataloader = FaceDetectionDataloader("data/valid", batch_size, (width, height))
for img, label in dataloader:
    # batch = 1 용 임시 squeeze
    img = img.squeeze()
    label = label.squeeze()

    _img = real2quantized(img, input_zero_point, input_scale)
    # _img = tf.cast(_img, input_dtype)
    _img = np.expand_dims(_img, axis=0).astype(input_dtype)

    model.set_tensor(input_details[0]["index"], _img)
    model.invoke()

    # output_data's shape: (batch x 12 x 12 x 2)
    # label's shape: (batch x detected x 4)
    output_data = model.get_tensor(output_details[0]["index"])
    output_data = quantized2real(output_data, output_zero_point, output_scale)


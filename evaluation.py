import os
import time
import argparse

from PIL import Image
import tensorflow as tf
import numpy as np

from read_mat_file import ground_truth
from dataloader import FaceDetectionDataloader, StaticTilingDataloader
from utils import real2quantized, quantized2real, idx2boundingbox, get_iou, merge_tile


def evaluate(bb_gt_collection, dataloader, iou_threshold, detect):
    # original code from https://github.com/nodefluxio/face-detector-benchmark/blob/master/wider_benchmark.py
    total_data = len(bb_gt_collection.keys())
    data_total_iou = 0
    data_total_precision = 0
    data_total_inference_time = 0

    # Evaluate face detector and iterate it over dataset
    for img, label, fn in dataloader:
        print(f"Image Shape: {img.shape}")
        # mode == common일 때 img의 shape는 (batch x 96 x 96 x 3)
        start_time = time.time()
        face_pred = detect(img) # face_pred's shape: (batch x 12 x 12 x 2)

        inf_time = time.time() - start_time
        data_total_inference_time += inf_time
        total_gt_face = len(label)

        ### Calc average IOU, Precision, and Average inferencing time ####
        total_iou = 0
        tp = 0
        pred_dict = dict()
        for l in label:
            max_iou_per_gt = 0

            for i, pred_bb in enumerate(face_pred):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                iou = get_iou(l, pred_bb)
                max_iou_per_gt = max(iou, max_iou_per_gt)

                # 찾은 face 개수 카운팅 -> precision 계산
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou += max_iou_per_gt

        if total_gt_face != 0:
            if len(pred_dict.keys()) > 0:
                for i in pred_dict:
                    if pred_dict[i] >= iou_threshold:
                        tp += 1
                precision = float(tp) / float(total_gt_face)

            else:
                precision = 0

            image_average_iou = total_iou / total_gt_face
            image_average_precision = precision

            data_total_iou += image_average_iou
            data_total_precision += image_average_precision

    result = dict()
    result['average_iou'] = float(data_total_iou) / float(total_data)
    result['mean_average_precision'] = float(data_total_precision) / float(
        total_data)
    result['average_inferencing_time'] = float(
        data_total_inference_time) / float(total_data)

    return result


def common_detect(img):
    _img = real2quantized(img, input_zero_point, input_scale)
    _img = tf.cast(_img, input_dtype)
    # _img = np.expand_dims(_img, axis=0).astype(input_dtype)

    model.set_tensor(input_details[0]["index"], _img[0])
    model.invoke()

    output_data = model.get_tensor(output_details[0]["index"])
    output_data = quantized2real(output_data, output_zero_point, output_scale)

    # if pred[1]:
    #     pred_bb = idx2boundingbox(i)
    # 현재 model output이 제대로 나오지 않아서 어떤 경우에 어떤 label으로 판별해야 하는지 알 수 없음(양자화 이슈로 추정)
    # 0: background, 1: face
    # ref: https://github.com/openmv/openmv/blob/master/src/lib/libtf/models/fomo_face_detection.txt
    pred_bbs = [idx2boundingbox(i, width, height,
                                output_width, output_height)
                for i, pred in enumerate(output_data) if pred[1]]
    # return output_data  # (batch x 12 x 12 x 2)
    return pred_bbs     # (x, y, w, h)의 좌표 형태


def tiling_detect(imgs):
    result = []
    for i, img in enumerate(imgs):
        detected = common_detect(img)
        pred_bbs = [merge_tile(i, pred_bb, tile_size, input_shape)
                    for pred_bb in detected]
        result.extend(pred_bbs)
    return result


model = tf.lite.Interpreter(model_path="./tflite/fomo_face_detection.tflite")
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

print(input_details, end="\n\n")
print(output_details, end="\n\n")

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
iou_threshold = 0.5
tile_size = (12, 8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["common", "static_tiling"])
    args = parser.parse_args()
    
    if args.mode == "common":
        dataloader = FaceDetectionDataloader("data/valid", batch_size, (width, height))
        detect = common_detect
    elif args.mode =="static_tiling":
        dataloader = StaticTilingDataloader("data/valid", batch_size, (width, height), (12, 8))
        detect = tiling_detect
    else:
        print("Invalid mode.")
        print("EXIT.")

    evaluate(ground_truth(), dataloader, iou_threshold, detect)
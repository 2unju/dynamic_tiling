import os
import time
import argparse
import json
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np

from read_mat_file import ground_truth
from dataloader import FaceDetectionDataloader, StaticTilingDataloader
from utils import real2quantized, quantized2real, idx2boundingbox, get_ious, merge_tile, _idx2boundingbox, resize_boundingbox


def draw_n_save(filename, bbx):
    img = Image.open(f"data/valid/{filename}")
    draw = ImageDraw.Draw(img)
    for box in bbx:
        if type(box) == tuple:
            box = list(box)
        box[2] += box[0]
        box[3] += box[1]
        _box = box
        box = [_box[1], _box[0], _box[3], _box[2]]
        draw.rectangle(box, outline=(255, 0, 0))
    img.save(f"result/draw/static/{filename}")


def evaluate(bb_gt_collection, dataloader, iou_threshold, detect):
    # original code from https://github.com/nodefluxio/face-detector-benchmark/blob/master/wider_benchmark.py
    total_data = len(bb_gt_collection.keys())
    data_total_iou = 0
    data_total_precision = 0
    data_total_inference_time = 0

    os.makedirs(f"result/draw/static", exist_ok=True)

    # Evaluate face detector and iterate it over dataset
    for img, label, fn, ts, _os in tqdm(dataloader):
        # print(f"Image Shape: {img.shape}")
        # mode == common일 때 img의 shape는 (batch x 96 x 96 x 3)
        
        start_time = time.time()
        if ts:
            face_pred = detect(img, ts, _os) # face_pred's shape: (batch x 12 x 12 x 2)
        else:
            face_pred = detect(img, _os)

        inf_time = time.time() - start_time
        data_total_inference_time += inf_time
        total_gt_face = len(label)

        if iou_threshold == 0.1:
            draw_n_save(fn[0], face_pred)

        ### Calc average IOU, Precision, and Average inferencing time ####
        total_iou = 0
        tp = 0
        pred_dict = dict()
        for l in label:
            max_iou_per_gt = 0

            for i, pred_bb in enumerate(face_pred):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                iou = get_ious(l, pred_bb)          # batch = 1이라 for문이 전체적으로 정상작동을 안함. 나중에 수정 필요
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
    result["recall"] = 0

    return result


def common_detect(img, origin_size): # img.shape = (batch x input_shape[0] x input_shape[1] x 3)
    _img = real2quantized(img, input_zero_point, input_scale)
    _img = tf.cast(_img, input_dtype)
    # _img = np.expand_dims(_img, axis=0).astype(input_dtype)

    model.set_tensor(input_details[0]["index"], _img)
    model.invoke()

    output_data = model.get_tensor(output_details[0]["index"])

    output_data = quantized2real(output_data, output_zero_point, output_scale)
    # if pred[1]:
    #     pred_bb = idx2boundingbox(i)
    # 현재 model output이 제대로 나오지 않아서 어떤 경우에 어떤 label으로 판별해야 하는지 알 수 없음(양자화 이슈로 추정)
    # 0: background, 1: face
    # ref: https://github.com/openmv/openmv/blob/master/src/lib/libtf/models/fomo_face_detection.txt

    pred_bbs = []
    for h_idx, preds in enumerate(output_data[0]):  # batch = 1이라(임시방편)
        for w_idx, pred in enumerate(preds):
            if pred[1] > iou_threshold:
                if origin_size:
                    # 96 x 96 -> 원본 이미지 사이즈로 리사이징 완료된 좌표값
                    pred_bbs.append(_idx2boundingbox(w_idx, h_idx, width, height, output_width, output_height, origin_size[0]))
                else:
                    # 96 x 96의 좌표값
                    pred_bbs.append(
                        idx2boundingbox(w_idx, h_idx, width, height, output_width, output_height))
    # pred_bbs = [
    #     [_idx2boundingbox(w_idx, h_idx, width, height, output_width, output_height)
    #     for w_idx, pred in enumerate(preds) if pred[1] > iou_threshold]
    #     for h_idx, preds in enumerate(output_data[0])       # batch = 1이라(임시방편)
    # ]
    return pred_bbs     # (x, y, w, h)의 좌표 형태


def tiling_detect(imgs, tile_size, origin_size):
    result = []
    for img, ts in zip(imgs, tile_size):
        # print(tile_size)      # 7 x 10
        # print(img[0].shape)   # 97 x 103 x 3
        # print(len(img))       # 70
        for i, cropped in enumerate(img):
            # tf.keras.utils.save_img("cropped_tmp.png", cropped)
            # _img = Image.open(f"cropped_tmp.png")
            # draw = ImageDraw.Draw(_img)
            # draw.rectangle([48, 0, 58, 10], outline=(255, 0, 0))
            # _img.save(f"cropped_draw_test.png")
            # exit()

            # 타일 크기 변경
            # cropped = tf.image.resize(cropped, [96, 96])
            _detected = common_detect(np.expand_dims(cropped, axis=0), None)
            # _detected = [resize_boundingbox(d, [96, 96], [284, 284]) for d in _detected]
            # 타일 크기 변경
            pred_bbs = [merge_tile(i, pred_bb, tile_size, [96, 96])
                        for pred_bb in _detected]
            # print(f"pred_bbs: {pred_bbs}")
            pred_bbs = [
                resize_boundingbox(pred_bb, [96 * tile_size[0], 96 * tile_size[1]], origin_size[0])
                for pred_bb in pred_bbs
            ]
            # print(f"resized: {pred_bbs}")
            # print(f"origin_shape: {origin_size}")
            # print()
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
iou_threshold = 0.1
tile_size = (2, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["common", "static_tiling"])
    args = parser.parse_args()
    
    if args.mode == "common":
        dataloader = FaceDetectionDataloader("data/valid", batch_size, (width, height))
        detect = common_detect
    elif args.mode =="static_tiling":
        dataloader = StaticTilingDataloader("data/valid", batch_size, (width, height), tile_size)
        detect = tiling_detect
    else:
        print("Invalid mode.")
        print("EXIT.")

    fig = plt.figure(figsize=(9, 6))

    mAP = []
    iou = []
    threshold = np.arange(0.1, 1, 0.1)
    result = dict()

    for t in threshold:
        res = evaluate(ground_truth(), dataloader, t, detect)

        mAP.append(res["mean_average_precision"])
        iou.append(res["average_iou"])
        result[t] = res

        print(json.dumps(res))

    # res = evaluate(ground_truth(), dataloader, iou_threshold, detect)
    os.makedirs(f"result/{args.mode}", exist_ok=True)
    with open(f"result/{args.mode}/{time.strftime('%Y%m%d%I%m', time.localtime())}.json", "w") as fp:
        json.dump(result, fp, ensure_ascii=False)

    print(json.dumps(result))
    ax = plt.axes(projection="3d")
    ax.plot3D(mAP, threshold, iou)
    ax.scatter3D(mAP, threshold, iou)

    ax.set_xlabel("mAP")
    ax.set_ylabel("Threshold")
    ax.set_zlabel("IoU")
    plt.savefig('result/_pr_curve.png')

    fig = plt.figure(figsize=(9, 6))
    plt.plot(mAP, threshold)
    plt.scatter(mAP, threshold)
    plt.xlabel("mAP")
    plt.ylabel("iou")
    plt.savefig('result/_pr_curve2.png')

import numpy as np
import matplotlib.pylab as plt
import scipy.io
import json


def ground_truth(path="data/eval_tools/ground_truth"):
    mat_file = scipy.io.loadmat(f"./{path}/wider_face_val.mat")

    gt_dict = dict()
    for label, boxes in zip(mat_file["file_list"], mat_file["face_bbx_list"]):
        for fn, box in zip(label[0], boxes[0]):
            box = box.squeeze().tolist()
            # box = [box[0]]    # Can't convert non-rectangular Python sequence to Tensor 에러를 해결하기 위한 임시조치
            # -> custom dataloader로 해결
            gt_dict[fn.squeeze().tolist()[0] + ".jpg"] = box
    return gt_dict

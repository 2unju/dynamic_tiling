import os
from PIL import Image
from tqdm import tqdm


def split_dataset():
    from read_mat_file import ground_truth

    split_point = 5
    many = []
    few = []

    gts = ground_truth()
    for gt in gts:
        bbx = gts[gt]
        if len(bbx) > split_point:
            many.append(gt)
        else:
            few.append(gt)

    with open("data/over_5_face.txt", "w") as fp:
        for filename in many: fp.write(f"{filename}\n")
    with open("data/under_5_face.txt", "w") as fp:
        for filename in few: fp.write(f"{filename}\n")


def merge_dataset():
    save_path = "data/valid"
    os.makedirs(save_path, exist_ok=True)

    base_path = "data/WIDER_val/images"
    dirlist = os.listdir(base_path)

    for dirname in dirlist:
        filelist = [fn for fn in os.listdir(os.path.join(base_path, dirname)) if fn.endswith(".jpg")]
        for filename in tqdm(filelist):
            img = Image.open(os.path.join(base_path, dirname, filename))
            img.save(os.path.join(save_path, filename))


def real2quantized(real_value, zero_point, scale):
    return (real_value / scale) + zero_point


def quantized2real(quantized_value, zero_point, scale):
    return (quantized_value - zero_point) * scale


def idx2boundingbox(w_idx, h_idx, width, height, output_width, output_height):
    w = width / output_width
    h = height / output_height
    x = w_idx * w
    y = h_idx * h
    return x, y, w, h


def get_centroid(bbx):
    center_width = bbx[0] + (bbx[2] / 2)
    center_height = bbx[1] + (bbx[3] / 2)
    return [center_width, center_height]


def resize_boundingbox(origin_bbx, origin_size, target_size):
    new_bbx = list(origin_bbx)
    origin_bbx = list(origin_bbx)
    for i in range(len(new_bbx)):
        new_bbx[i] = origin_bbx[i] / origin_size[i % 2] * target_size[(i + 1) % 2]
    return new_bbx


def _idx2boundingbox(w_idx, h_idx, width, height, output_width, output_height, origin_size):
    w = width / output_width
    h = height / output_height
    x = w_idx * w
    y = h_idx * h
    
    x = x / width * origin_size[1]
    y = y / height * origin_size[0]
    w = w / width * origin_size[1]
    h = h / height * origin_size[0]
    return x, y, w, h


def merge_tile(idx, origin_bb, tile_size, input_shape):
    new_bb = list(origin_bb)
    n, m = tile_size
    img_width, img_height = input_shape
    new_bb[0] += int(idx % n) * img_width
    new_bb[1] += int(idx // n) * img_height

    return new_bb


def isin(center, bbx):
    if (center[0] >= bbx[0] and center[0] <= bbx[0] + bbx[2]) and \
        (center[1] >= bbx[1] and center[1] <= bbx[1] + bbx[3]):
        return 1
    return 0


def get_iou_n_tp(boxes, pred):
    iou = 0.0
    tp = 0
    center = get_centroid(pred)

    for box in boxes:
        _iou = get_iou(pred, box)
        iou = max(iou, _iou)
        if not tp and isin(center, box):
            tp = 1
    return iou, tp


def get_iou(boxA, boxB):
    # code from https://github.com/nodefluxio/face-detector-benchmark/blob/master/wider_benchmark.py
    """
	Calculate the Intersection over Union (IoU) of two bounding boxes.
	Parameters
	----------
	boxA = np.array( [ xmin,ymin,xmax,ymax ] )
	boxB = np.array( [ xmin,ymin,xmax,ymax ] )
	Returns
	-------
	float
		in [0, 1]
	"""

    bb1 = dict()
    bb1['x1'] = boxA[0]
    bb1['y1'] = boxA[1]
    bb1['x2'] = boxA[2]
    bb1['y2'] = boxA[3]

    bb2 = dict()
    bb2['x1'] = boxB[0]
    bb2['y1'] = boxB[1]
    bb2['x2'] = boxB[2]
    bb2['y2'] = boxB[3]

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes area
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou

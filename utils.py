import os
from PIL import Image
from tqdm import tqdm


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


def idx2boundingbox(idx, width, height, output_width, output_height):
    x = (idx % int(width / output_width)) * output_width
    y = (idx // int(height / output_height)) * output_height
    w = output_width
    h = output_height
    return x, y, w, h


def merge_tile(idx, origin_bb, tile_size, input_shape):
    n, m = tile_size
    img_width, img_height = input_shape
    origin_bb[0] += int(idx % (img_width / n)) * n
    origin_bb[1] += int(idx // (img_height / m)) * m
    return origin_bb


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

import numpy as np

def scale_boxes(boxes, orig_w, orig_h, input_size=(640, 640)):
    sx, sy = orig_w / input_size[0], orig_h / input_size[1]
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return boxes.astype(int)

def validate_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    return x1, y1, x2, y2

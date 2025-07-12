import cv2
import numpy as np

def nms(boxes, scores, conf_thres, iou_thres):
    mask = scores > conf_thres
    if not mask.any():
        return []
    boxes = boxes[mask]
    scores = scores[mask]
    idxs = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_thres, iou_thres
    )
    if isinstance(idxs, np.ndarray):
        idxs = idxs.flatten()
    return np.array(idxs, dtype=int)

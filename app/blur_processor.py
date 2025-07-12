import cv2
import numpy as np

class BlurProcessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def blur(self, frame, bboxes, kernel=None):
        for (x1, y1, x2, y2) in bboxes:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            roi = frame[y1:y2, x1:x2]
            if roi.size:
                k = kernel or 99
                blurred = cv2.GaussianBlur(roi, (k, k), 30)
                frame[y1:y2, x1:x2] = blurred
        return frame

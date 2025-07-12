from .onnx_model import OnnxModel
from ..nms import nms
from ..bbox_utils import scale_boxes

class FaceDetector(OnnxModel):
    def detect(self, img, conf=0.25, iou=0.45):
        outs, w, h = self(img)
        preds = outs[0][0].T                # (8400,5)
        boxes, scores = preds[:, :4], preds[:, 4]
        idxs = nms(boxes, scores, conf, iou)
        boxes = scale_boxes(boxes[idxs], w, h, self.in_sz)
        return boxes, scores[idxs]

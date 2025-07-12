from .onnx_model import OnnxModel
from ..nms import nms
from ..bbox_utils import scale_boxes

class PlateDetector(OnnxModel):
    def detect(self, img, conf=0.1, iou=0.45):
        outs, w, h = self(img)
        preds = outs[0][0].T                # (N,8+)  [x,y,x,y,conf,cls,text,text_conf...]
        boxes, scores = preds[:, :4], preds[:, 4]
        idxs = nms(boxes, scores, conf, iou)
        boxes = scale_boxes(boxes[idxs], w, h, self.in_sz)
        texts = preds[idxs, 6] if preds.shape[1] > 6 else [""]*len(idxs)
        text_confs = preds[idxs, 7] if preds.shape[1] > 7 else [1.0]*len(idxs)
        return boxes, texts, text_confs

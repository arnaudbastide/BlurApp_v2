import numpy as np
import cv2
import onnxruntime as ort

class OnnxModel:
    def __init__(self, path, input_size=(640, 640), providers=None):
        sess = ort.SessionOptions()
        sess.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            path, sess, providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.in_name = self.sess.get_inputs()[0].name
        self.in_sz = input_size

    def __call__(self, img_bgr):
        h, w = img_bgr.shape[:2]
        img = cv2.resize(img_bgr, self.in_sz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = img.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        return self.sess.run(None, {self.in_name: tensor.astype(np.float16)}), w, h

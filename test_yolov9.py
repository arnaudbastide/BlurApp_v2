import os
os.environ["ORT_DISABLE_TENSORRT"] = "1"

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, List, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort

# Load CUDA dependencies
try:
    ort.preload_dlls(cuda=True, cudnn=True)
    import ctypes
    ctypes.WinDLL("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/cublas64_12.dll")
    ctypes.WinDLL("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvrtc64_120_0.dll")
except Exception as e:
    print(f"Warning: Failed to load CUDA dependencies: {e}. Falling back to CPU.")

@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass(frozen=True)
class DetectionResult:
    bounding_box: BoundingBox
    confidence: float
    class_name: str

class BaseDetector:
    def predict(self, frame: np.ndarray) -> List[DetectionResult]:
        raise NotImplementedError

class DefaultDetector(BaseDetector):
    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        class_name: str,
        conf_thresh: float = 0.3,
        iou_thresh: float = 0.7,
        input_size: tuple[int, int] = (640, 640),
        providers: Optional[Sequence[Union[str, tuple[str, dict]]]] = None,
        sess_options: Optional[ort.SessionOptions] = None,
        person_class_idx: int = 0,  # Class index for "person" in yolov9c
    ) -> None:
        self.model_path = model_path
        self.class_name = class_name
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.person_class_idx = person_class_idx
        
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                providers=providers,
                sess_options=sess_options
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize ONNX session for {model_path}: {e}")
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"{self.class_name.capitalize()} detector initialized with model: {model_path}")
        print(f"Input name: {self.input_name}")
        print(f"Output names: {self.output_names}")
        print(f"Available providers: {ort.get_available_providers()}")

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, int, int]:
        h, w = frame.shape[:2]
        target_w, target_h = self.input_size
        x_scale = w / target_w
        y_scale = h / target_h

        resized = cv2.resize(frame, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, 0)
        return tensor, x_scale, y_scale, w, h

    def postprocess(
        self,
        outputs: List[np.ndarray],
        x_scale: float,
        y_scale: float,
        img_width: int,
        img_height: int,
        output_index: int = 0,
        box_format: str = "center"
    ) -> List[DetectionResult]:
        predictions = outputs[output_index]
        print(f"Processing {self.class_name} output {self.output_names[output_index]}, shape: {predictions.shape}")
        
        if len(predictions.shape) != 3:
            print(f"Skipping invalid {self.class_name} output shape: {predictions.shape}")
            return []
        
        # Handle face model (1, 5, 8400) or person model (1, 84, 8400)
        if predictions.shape[1] == 5:  # Face model
            predictions = np.transpose(predictions, (0, 2, 1))  # (1, 8400, 5)
            confidences = predictions[0, :, 4]
            boxes = predictions[0, :, :4]
        elif predictions.shape[1] == 84:  # Person model
            predictions = np.transpose(predictions, (0, 2, 1))  # (1, 8400, 84)
            confidences = predictions[0, :, 4 + self.person_class_idx]  # Person class score
            boxes = predictions[0, :, :4]
        else:
            print(f"Skipping invalid {self.class_name} output shape: {predictions.shape}")
            return []
        
        print(f"{self.class_name.capitalize()} Max confidence: {confidences.max():.4f}, Min confidence: {confidences.min():.4f}")
        print(f"{self.class_name.capitalize()} Sample predictions (first 5): {predictions[0, :5]}")
        
        mask = confidences > self.conf_thresh
        filtered = boxes[mask]
        filtered_confidences = confidences[mask]
        
        print(f"{self.class_name.capitalize()} Filtered predictions: {len(filtered)} > {self.conf_thresh}")
        
        if filtered.size == 0:
            return []
        
        scores = filtered_confidences.tolist()
        
        if box_format == "center":
            coords = np.zeros_like(filtered)
            coords[:, 0] = filtered[:, 0] - filtered[:, 2] / 2
            coords[:, 1] = filtered[:, 1] - filtered[:, 3] / 2
            coords[:, 2] = filtered[:, 0] + filtered[:, 2] / 2
            coords[:, 3] = filtered[:, 1] + filtered[:, 3] / 2
        else:
            coords = filtered.copy()
        
        coords[:, 0] = np.clip(coords[:, 0] * x_scale, 0, img_width)
        coords[:, 1] = np.clip(coords[:, 1] * y_scale, 0, img_height)
        coords[:, 2] = np.clip(coords[:, 2] * x_scale, 0, img_width)
        coords[:, 3] = np.clip(coords[:, 3] * y_scale, 0, img_height)
        
        print(f"{self.class_name.capitalize()} Sample scaled boxes: {coords[:5]}")
        
        valid_boxes = []
        valid_scores = []
        for i, (x1, y1, x2, y2) in enumerate(coords):
            w = x2 - x1
            h = y2 - y1
            min_size = 10
            max_size = min(img_width, img_height) / 2
            if w < min_size or h < min_size or w > max_size or h > max_size:
                print(f"Skipping invalid {self.class_name} box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                continue
            valid_boxes.append([x1, y1, x2, y2])
            valid_scores.append(scores[i])
        
        if not valid_boxes:
            return []
        
        idxs = cv2.dnn.NMSBoxes(
            valid_boxes,
            valid_scores,
            self.conf_thresh,
            self.iou_thresh
        )
        
        if isinstance(idxs, np.ndarray):
            idxs = idxs.flatten()
        elif isinstance(idxs, list):
            idxs = [i[0] if isinstance(i, (list, tuple)) else i for i in idxs]
        
        print(f"{self.class_name.capitalize()} NMS kept {len(idxs)} detections")
        detections: List[DetectionResult] = []
        for i in idxs:
            x1, y1, x2, y2 = map(int, valid_boxes[i])
            if x2 <= x1 or y2 <= y1:
                print(f"Skipping invalid {self.class_name} box: ({x1}, {y1}, {x2}, {y2})")
                continue
            
            detections.append(
                DetectionResult(
                    bounding_box=BoundingBox(x1, y1, x2, y2),
                    confidence=float(valid_scores[i]),
                    class_name=self.class_name
                )
            )
        
        return detections

    def predict(self, frame: np.ndarray) -> List[DetectionResult]:
        tensor, x_scale, y_scale, img_width, img_height = self.preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        print(f"{self.class_name.capitalize()} Output shapes: {[o.shape for o in outputs]}")
        
        for output_index in range(len(self.output_names)):
            for box_format in ["center", "corner"]:
                print(f"Trying {self.class_name} output {self.output_names[output_index]} with box format {box_format}")
                dets = self.postprocess(outputs, x_scale, y_scale, img_width, img_height, output_index, box_format)
                if dets:
                    print(f"Found {len(dets)} {self.class_name} detections with output {self.output_names[output_index]} and format {box_format}")
                    return dets
        print(f"No valid {self.class_name} detections found")
        return []

class DualDetector:
    def __init__(
        self,
        face_model_path: Union[str, os.PathLike],
        person_model_path: Union[str, os.PathLike],
        face_conf_thresh: float = 0.3,
        person_conf_thresh: float = 0.3,
        face_iou_thresh: float = 0.7,
        person_iou_thresh: float = 0.7,
        input_size: tuple[int, int] = (640, 640),
        providers: Optional[Sequence[Union[str, tuple[str, dict]]]] = None,
        sess_options: Optional[ort.SessionOptions] = None,
    ):
        self.face_detector = DefaultDetector(
            model_path=face_model_path,
            class_name="face",
            conf_thresh=face_conf_thresh,
            iou_thresh=face_iou_thresh,
            input_size=input_size,
            providers=providers,
            sess_options=sess_options,
        )
        self.person_detector = DefaultDetector(
            model_path=person_model_path,
            class_name="person",
            conf_thresh=person_conf_thresh,
            iou_thresh=person_iou_thresh,
            input_size=input_size,
            providers=providers,
            sess_options=sess_options,
            person_class_idx=0,  # Adjust if "person" class index differs
        )

    def predict(self, frame: Union[np.ndarray, str]) -> List[DetectionResult]:
        if isinstance(frame, str):
            img = cv2.imread(frame)
            if img is None:
                raise ValueError(f"Cannot read image: {frame}")
            frame = img
        
        face_detections = self.face_detector.predict(frame)
        person_detections = self.person_detector.predict(frame)
        return face_detections + person_detections

    def draw_predictions(self, frame: Union[np.ndarray, str], draw_raw: bool = False) -> np.ndarray:
        if isinstance(frame, str):
            img = cv2.imread(frame)
            if img is None:
                raise ValueError(f"Cannot read image: {frame}")
            frame = img
        
        dets = self.predict(frame)
        
        if draw_raw:
            for detector, color in [(self.face_detector, (0, 0, 255)), (self.person_detector, (255, 0, 0))]:
                tensor, x_scale, y_scale, img_width, img_height = detector.preprocess(frame)
                outputs = detector.session.run(detector.output_names, {detector.input_name: tensor})
                for output_index in range(len(outputs)):
                    predictions = outputs[output_index]
                    if len(predictions.shape) != 3 or predictions.shape[1] not in [5, 84]:
                        continue
                    predictions = np.transpose(predictions, (0, 2, 1))
                    confidences = predictions[0, :, 4] if predictions.shape[2] == 5 else predictions[0, :, 4 + detector.person_class_idx]
                    mask = confidences > detector.conf_thresh
                    filtered = predictions[0, :, :4][mask]
                    if filtered.size == 0:
                        continue
                    for box, box_format in [(filtered, "center"), (filtered, "corner")]:
                        for b in box:
                            if box_format == "center":
                                x1 = (b[0] - b[2] / 2) * x_scale
                                y1 = (b[1] - b[3] / 2) * y_scale
                                x2 = (b[0] + b[2] / 2) * x_scale
                                y2 = (b[1] + b[3] / 2) * y_scale
                            else:
                                x1, y1, x2, y2 = b * [x_scale, y_scale, x_scale, y_scale]
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            if x2 <= x1 or y2 <= y1:
                                continue
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        for det in dets:
            bb = det.bounding_box
            color = (0, 255, 0) if det.class_name == "face" else (255, 0, 0)
            cv2.rectangle(frame, (bb.x1, bb.y1), (bb.x2, bb.y2), color, 2)
            text = f"{det.class_name.capitalize()} {det.confidence * 100:.1f}%"
            cv2.putText(frame, text, (bb.x1, bb.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (bb.x1, bb.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame

def process_video(
    face_model_path: str,
    person_model_path: str,
    video_path: str,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    face_conf_thresh: float = 0.3,
    person_conf_thresh: float = 0.3,
    face_iou_thresh: float = 0.7,
    person_iou_thresh: float = 0.7
):
    detector = DualDetector(
        face_model_path=face_model_path,
        person_model_path=person_model_path,
        face_conf_thresh=face_conf_thresh,
        person_conf_thresh=person_conf_thresh,
        face_iou_thresh=face_iou_thresh,
        person_iou_thresh=person_iou_thresh
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("Warning: mp4v codec failed. Trying XVID.")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise ValueError("Cannot initialize video writer")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        print(f"Frame {count}/{total}")
        
        out = detector.draw_predictions(frame, draw_raw=(count % 10 == 0))
        
        if count % 10 == 0:
            cv2.imwrite(f"debug_frame_{count}.jpg", out)
        
        if writer:
            writer.write(out)
        else:
            cv2.imshow("Face and Person Detection", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if max_frames and count >= max_frames:
            break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {count} frames")

def test_single_frame(
    face_model_path: str,
    person_model_path: str,
    image_path: str,
    output_path: Optional[str] = None
):
    detector = DualDetector(face_model_path=face_model_path, person_model_path=person_model_path)
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    result = detector.draw_predictions(frame, draw_raw=True)
    
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Saved result to: {output_path}")
    else:
        cv2.imshow("Face and Person Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_model_path = "yolov9e-face.onnx"
    person_model_path = "yolov9c.onnx"
    video_path = "street.mp4"
    output_path = "output_video.mp4"
    try:
        #test_single_frame(face_model_path, person_model_path, "frame.jpg", "out_frame.jpg")
        process_video(
            face_model_path,
            person_model_path,
            video_path,
            None,
            max_frames=None,
            face_conf_thresh=0.1,
            person_conf_thresh=0.2,
            face_iou_thresh=0.9,
            person_iou_thresh=0.8
        )
    except Exception as e:
        print(f"Error: {e}")
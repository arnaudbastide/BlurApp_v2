import os
os.environ["ORT_DISABLE_TENSORRT"] = "1"

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, List, Optional, Union

import time
import cv2
import numpy as np
import onnxruntime as ort

# Ensure CUDA dependencies are loaded (Windows-specific)
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
class FaceDetectionResult:
    bounding_box: BoundingBox
    confidence: float
    class_name: str = "face"

class BaseFaceDetector:
    def predict(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        raise NotImplementedError

class DefaultFaceDetector(BaseFaceDetector):
    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        conf_thresh: float = 0.15,  # Lowered for more detections
        iou_thresh: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        providers: Optional[Sequence[Union[str, tuple[str, dict]]]] = None,
        sess_options: Optional[ort.SessionOptions] = None,
    ) -> None:
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        
        # Set default providers
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        # Set session options
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Initialize ONNX session
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                providers=providers,
                sess_options=sess_options
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize ONNX session: {e}")
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"Face detector initialized with model: {model_path}")
        print(f"Input name: {self.input_name}")
        print(f"Output names: {self.output_names}")
        print(f"Available providers: {ort.get_available_providers()}")

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Preprocess frame for model input."""
        h, w = frame.shape[:2]
        target_w, target_h = self.input_size
        x_scale = w / target_w
        y_scale = h / target_h

        resized = cv2.resize(frame, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, 0)
        return tensor, x_scale, y_scale

    def postprocess(
        self,
        outputs: List[np.ndarray],
        x_scale: float,
        y_scale: float,
        output_index: int = 0,
        box_format: str = "center"  # "center" for [cx, cy, w, h], "corner" for [x1, y1, x2, y2]
    ) -> List[FaceDetectionResult]:
        """Postprocess model outputs to get face detections."""
        predictions = outputs[output_index]
        print(f"Processing output {self.output_names[output_index]}, shape: {predictions.shape}")
        
        # Transpose (1, 5, 8400) to (1, 8400, 5)
        predictions = np.transpose(predictions, (0, 2, 1))
        
        detections: List[FaceDetectionResult] = []
        for dets in predictions:  # Shape: (8400, 5)
            # Debug raw predictions
            print(f"Sample predictions (first 100): {dets[:100]}")
            
            # Filter by confidence
            mask = dets[:, 4] > self.conf_thresh
            filtered = dets[mask]
            print(f"Filtered predictions: {len(filtered)} > {self.conf_thresh}")
            
            if filtered.size == 0:
                continue
            
            boxes = filtered[:, :4]
            scores = filtered[:, 4].tolist()
            
            # Convert box format
            if box_format == "center":
                # [x_center, y_center, w, h] to [x1, y1, x2, y2]
                coords = np.zeros_like(boxes)
                coords[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
                coords[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
                coords[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
                coords[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
            else:
                # Assume [x1, y1, x2, y2]
                coords = boxes.copy()
            
            # Scale to original image size
            coords[:, 0] *= x_scale
            coords[:, 1] *= y_scale
            coords[:, 2] *= x_scale
            coords[:, 3] *= y_scale
            
            print(f"Sample scaled boxes: {coords[:100]}")
            
            # Apply NMS
            idxs = cv2.dnn.NMSBoxes(
                coords.tolist(),
                scores,
                self.conf_thresh,
                self.iou_thresh
            )
            
            if isinstance(idxs, np.ndarray):
                idxs = idxs.flatten()
            elif isinstance(idxs, list):
                idxs = [i[0] if isinstance(i, (list, tuple)) else i for i in idxs]
            
            print(f"NMS kept {len(idxs)} detections")
            for i in idxs:
                x1, y1, x2, y2 = map(int, coords[i])
                if x2 <= x1 or y2 <= y1:
                    print(f"Skipping invalid box: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                detections.append(
                    FaceDetectionResult(
                        bounding_box=BoundingBox(x1, y1, x2, y2),
                        confidence=float(scores[i])
                    )
                )
        
        return detections

    def predict(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """Predict faces in a frame."""
        tensor, x_scale, y_scale = self.preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        print(f"Output shapes: {[o.shape for o in outputs]}")
        
        # Try both outputs and box formats
        for output_index, output_name in enumerate(self.output_names):
            for box_format in ["center", "corner"]:
                print(f"Trying output {output_name} with box format {box_format}")
                dets = self.postprocess(outputs, x_scale, y_scale, output_index, box_format)
                if dets:
                    print(f"Found {len(dets)} detections with output {output_name} and format {box_format}")
                    return dets
        print("No valid detections found")
        return []

class FaceDetector:
    def __init__(
        self,
        detector: Optional[BaseFaceDetector] = None,
        model_path: Union[str, os.PathLike] = None,
        conf_thresh: float = 0.15,
        iou_thresh: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        providers: Optional[Sequence[Union[str, tuple[str, dict]]]] = None,
        sess_options: Optional[ort.SessionOptions] = None,
    ):
        if detector:
            self.detector = detector
        elif model_path:
            self.detector = DefaultFaceDetector(
                model_path=model_path,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                input_size=input_size,
                providers=providers,
                sess_options=sess_options,
            )
        else:
            raise ValueError("Provide either detector or model_path")

    def predict(self, frame: Union[np.ndarray, str]) -> List[FaceDetectionResult]:
        if isinstance(frame, str):
            img = cv2.imread(frame)
            if img is None:
                raise ValueError(f"Cannot read image: {frame}")
            frame = img
        return self.detector.predict(frame)

    def draw_predictions(self, frame: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(frame, str):
            img = cv2.imread(frame)
            if img is None:
                raise ValueError(f"Cannot read image: {frame}")
            frame = img
        for det in self.predict(frame):
            bb = det.bounding_box
            cv2.rectangle(frame, (bb.x1, bb.y1), (bb.x2, bb.y2), (0, 255, 0), 2)
            text = f"Face {det.confidence * 100:.1f}%"
            cv2.putText(frame, text, (bb.x1, bb.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (bb.x1, bb.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

def process_video(
    model_path: str,
    video_path: str,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    conf_thresh: float = 0.15,
    iou_thresh: float = 0.45
):
    """Process a video for face detection and optionally save the output."""
    detector = FaceDetector(model_path=model_path, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'XVID' if mp4v fails
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("Warning: Video writer failed. Trying XVID codec.")
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
        #time.sleep(0.2)
        print(f"Frame {count}/{total}")
        
        out = detector.draw_predictions(frame)
        
        if writer:
            writer.write(out)
        else:
            cv2.imshow("Face Detection", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if max_frames and count >= max_frames:
            break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {count} frames")

def test_single_frame(model_path: str, image_path: str, output_path: Optional[str] = None):
    """Test face detection on a single image."""
    detector = FaceDetector(model_path=model_path)
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    result = detector.draw_predictions(frame)
    
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Saved result to: {output_path}")
    else:
        cv2.imshow("Face Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_model_path = "yolov9e-face.onnx"
    person_model_path = "yolov9c.onnx"
    video_path = "D113_Alsace_Pomme.mp4"
    output_path = "output_video.mp4"
    try:
        # Test single frame first
        #test_single_frame(model_path, "frame.jpg", "out_frame.jpg")
        # Process video with limited frames for debugging
        process_video(face_model_path, video_path, None, max_frames=None, conf_thresh=0.1, iou_thresh=0.8)
    except Exception as e:
        print(f"Error: {e}")
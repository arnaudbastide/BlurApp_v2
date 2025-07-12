import argparse
import os
import sys
from pathlib import Path
import logging
import queue
import threading
import time
from typing import Optional, Tuple, List, Dict, Generator
from dataclasses import dataclass
from contextlib import contextmanager
from collections import deque

# Enhanced logging with colors
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[35m'
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(level: str = "DEBUG", log_file: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))
        logger.addHandler(file_handler)

# Performance monitoring
class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            'frame_times': deque(maxlen=window_size),
            'inference_times': deque(maxlen=window_size),
            'blur_times': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'gpu_memory': deque(maxlen=window_size) if torch.cuda.is_available() else None
        }
        self.lock = threading.Lock()

    def record_frame_time(self, time_ms: float):
        with self.lock:
            self.metrics['frame_times'].append(time_ms)

    def record_inference_time(self, time_ms: float):
        with self.lock:
            self.metrics['inference_times'].append(time_ms)

    def record_blur_time(self, time_ms: float):
        with self.lock:
            self.metrics['blur_times'].append(time_ms)

    def record_memory_usage(self):
        with self.lock:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.metrics['memory_usage'].append(memory_mb)
            if self.metrics['gpu_memory'] is not None and torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                self.metrics['gpu_memory'].append(gpu_memory_mb)

    def get_stats(self) -> Dict:
        with self.lock:
            stats = {}
            for metric, values in self.metrics.items():
                if values and (metric != 'gpu_memory' or values):
                    stats[metric] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            if 'frame_times' in stats and stats['frame_times']['avg'] > 0:
                stats['fps'] = 1000.0 / stats['frame_times']['avg']
            return stats

# CUDA setup
def setup_cuda_environment():
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]
    if os.name == 'nt':
        for cuda_path in cuda_paths:
            if os.path.isdir(cuda_path):
                os.add_dll_directory(cuda_path)
                logging.info(f"Added CUDA DLL directory: {cuda_path}")
                return True
        logging.warning("No CUDA installation found. GPU acceleration may not work.")
        return False
    return True

CUDA_AVAILABLE = setup_cuda_environment()

import numpy as np
import cv2

print(cv2.getBuildInformation())

import torch
import psutil
import gc
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from deep_sort_realtime.deepsort_tracker import DeepSort

from PIL import Image

# Configuration with validation
@dataclass
class DetectionConfig:
    object_model_path: str
    face_model_path: str
    output_path: Optional[str] = None
    object_confidence_threshold: float = 0.25
    face_confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    face_match_iou_threshold: float = 0.45
    face_region_ratio: float = 0.25
    face_width_reduction: float = 0.15
    draw_detections: bool = True
    blur_objects: bool = False
    blur_faces: bool = True
    blur_strength: int = 21
    blur_expansion: float = 2.0
    min_ellipse_size: int = 5
    max_detections: int = 200
    input_size: int = 640
    device: str = "0"
    deepsort_max_age: int = 10
    deepsort_min_hits: int = 3
    deepsort_iou_tracker: float = 0.15
    enable_cuda_decoder: bool = True
    enable_cuda_encoder: bool = True
    output_codec: str = "h264"
    output_bitrate: int = 5000000
    skip_frames: int = 1
    enable_async_processing: bool = False
    buffer_size: int = 10
    classes: List[int] = None
    no_blur_track_ids: List[int] = None
    track_id_font_scale: float = 1.0
    track_id_font_thickness: int = 3
    track_id_bg_alpha: float = 0.85  # Increased for better visibility

    def __post_init__(self):
        if self.classes is None:
            self.classes = [0, 2, 3, 5, 7]
        if self.no_blur_track_ids is None:
            self.no_blur_track_ids = []
        self.validate()

    def validate(self):
        errors = []
        for model_path in [self.object_model_path, self.face_model_path]:
            if not Path(model_path).exists():
                errors.append(f"Model file not found: {model_path}")
        if not 0.0 <= self.object_confidence_threshold <= 1.0:
            errors.append("object_confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.face_confidence_threshold <= 1.0:
            errors.append("face_confidence_threshold must be between 0.0 and 1.0")
        if self.blur_strength < 3 or self.blur_strength % 2 == 0:
            errors.append("blur_strength must be odd and >= 3")
        if self.blur_expansion < 0.1:
            errors.append("blur_expansion must be >= 0.1")
        if self.skip_frames < 1:
            errors.append("skip_frames must be >= 1")
        if self.output_path and Path(self.output_path).suffix.lower() not in {'.mp4', '.avi'}:
            errors.append("output_path must have .mp4 or .avi extension")
        if not 0.1 <= self.track_id_bg_alpha <= 1.0:
            errors.append("track_id_bg_alpha must be between 0.1 and 1.0")
        if self.track_id_font_scale < 0.5:
            errors.append("track_id_font_scale must be >= 0.5")
        if self.track_id_font_thickness < 1:
            errors.append("track_id_font_thickness must be >= 1")
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

# Async frame buffer
class AsyncFrameBuffer:
    def __init__(self, maxsize: int = 10):
        self.queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def put(self, item: Tuple, timeout: float = 1.0) -> bool:
        try:
            self.queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout: float = 1.0) -> Optional[Tuple]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def clear(self):
        with self.lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

# Letterbox function (CPU version)
def letterbox(img: np.ndarray, new_shape: tuple = (640, 640), color: tuple = (114, 114, 114)) -> Tuple[np.ndarray, tuple, tuple]:
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def calculate_intersection_area(box1: List[float], box2: List[float]) -> float:
    """Calculate the intersection area between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there's no intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2 - x1) * (y2 - y1)
    return intersection_area

def calculate_containment_ratio(face_bbox: List[float], object_bbox: List[float]) -> float:
    """Calculate how much of the face is contained within the object bbox."""
    intersection_area = calculate_intersection_area(face_bbox, object_bbox)
    face_area = (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1])
    return intersection_area / face_area if face_area > 0 else 0.0

def is_face_inside_object(face_bbox: List[float], object_bbox: List[float]) -> bool:
    """Check if face bounding box is completely inside object bounding box."""
    return (face_bbox[0] >= object_bbox[0] and  # face left >= object left
            face_bbox[1] >= object_bbox[1] and  # face top >= object top  
            face_bbox[2] <= object_bbox[2] and  # face right <= object right
            face_bbox[3] <= object_bbox[3])     # face bottom <= object bottom

# Calculate IoU
def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

# Custom NMS
def custom_nms(detections: List[List[float]], iou_threshold: float = 0.4) -> List[List[float]]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: (x[5], x[4]), reverse=True)
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        remaining = []
        for det in detections:
            iou = calculate_iou(current[:4], det[:4])
            threshold = iou_threshold
            if (current[5] == 0 and det[5] == 1) or (current[5] == 1 and det[5] == 0):
                threshold = 0.1
            if iou <= threshold:
                remaining.append(det)
        detections = remaining
    return keep

# Updated Decoder
class OptimizedCUDADecoder:
    def __init__(self, video_path: str, use_cuda: bool = True, buffer_size: int = 5):
        self.video_path = Path(video_path)
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.buffer_size = buffer_size
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.total_frames: int = 0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.gpu_frame: Optional[cv2.cuda.GpuMat] = None
        self.gpu_temp: Optional[cv2.cuda.GpuMat] = None
        self.hw_acceleration: bool = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_decoder()
        if self.use_cuda:
            self._initialize_gpu_memory()

    def _initialize_decoder(self):
        try:
            if not self.video_path.exists():
                raise FileNotFoundError(f"Video file not found: {self.video_path}")
            backends = [cv2.CAP_FFMPEG, cv2.CAP_OPENCV_MJPEG, cv2.CAP_ANY]
            for backend in backends:
                self.cap = cv2.VideoCapture(str(self.video_path), backend)
                if self.use_cuda and backend == cv2.CAP_FFMPEG:
                    try:
                        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                        os.environ["OPENCV_FFMPEG_CAPTURE"] = "video_codec;h264_cuvid"
                        self.hw_acceleration = True
                        self.logger.error(f"Hardware acceleration enabled for decoding")
                    except Exception as e:
                        self.logger.error(f"Hardware acceleration setup failed: {e}")
                if self.cap.isOpened():
                    self.logger.error(f"Video opened with backend: {backend}, HW accel: {self.hw_acceleration}")
                    break
                self.cap.release()
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            self.fps = max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"Video properties: {self.width}x{self.height} @ {self.fps:.2f}fps, {self.total_frames} frames")
            if self.width <= 0 or self.height <= 0:
                raise ValueError("Invalid video dimensions")
            if self.total_frames <= 0:
                self.logger.warning("Could not determine total frame count")
                self.total_frames = -1
        except Exception as e:
            self.logger.error(f"Decoder initialization failed: {e}")
            self.release()
            raise

    def _initialize_gpu_memory(self):
        try:
            self.gpu_frame = cv2.cuda.GpuMat()
            self.gpu_temp = cv2.cuda.GpuMat()
            self.logger.info("GPU memory allocated successfully")
        except Exception as e:
            self.logger.warning(f"GPU memory allocation failed: {e}")
            self.use_cuda = False

    def read_and_preprocess(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            if not self.cap or not self.cap.isOpened():
                return False, None
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                if frame is None or frame.size == 0:
                    return False, None
                return True, frame
            return False, None
        except Exception as e:
            self.logger.error(f"Error reading frame {self.frame_count}: {e}")
            return False, None

    def get_frame_size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def get_progress(self) -> float:
        if self.total_frames > 0:
            return min(100.0, (self.frame_count / self.total_frames) * 100)
        return 0.0

    def letterbox_gpu(self, frame: np.ndarray, new_shape: tuple) -> Tuple[np.ndarray, tuple, tuple]:
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        shape = frame.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = (r, r)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw / 2, dh / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        if self.use_cuda:
            try:
                self.gpu_frame.upload(frame)
                self.gpu_temp = cv2.cuda.resize(self.gpu_frame, new_unpad, interpolation=cv2.INTER_LINEAR)
                img_gpu = cv2.cuda.copyMakeBorder(
                    self.gpu_temp, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                img = img_gpu.download()
                self.logger.debug("Performed letterbox on GPU")
                return img, ratio, (dw, dh)
            except Exception as e:
                self.logger.warning(f"CUDA letterbox failed: {e}")
                self.use_cuda = False
        
        img, ratio, (dw, dh) = letterbox(frame, new_shape)
        return img, ratio, (dw, dh)

    def release(self):
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.gpu_frame:
                del self.gpu_frame
                self.gpu_frame = None
            if self.gpu_temp:
                del self.gpu_temp
                self.gpu_temp = None
            self.logger.info("Decoder released successfully")
        except Exception as e:
            self.logger.error(f"Error releasing decoder: {e}")
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

# Updated Encoder with CUDA
class OptimizedCUDAEncoder:
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], 
                 use_cuda: bool = True, codec: str = 'h264', bitrate: int = 5000000):
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.codec = codec.lower()
        self.bitrate = bitrate
        self.writer: Optional[cv2.VideoWriter] = None
        self.frames_written: int = 0
        self.gpu_frame: Optional[cv2.cuda.GpuMat] = None
        self.hw_acc = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_encoder()
        if self.use_cuda:
            self._initialize_gpu_memory()

    CODEC_CONFIGS = {
        'h264': ['H264', 'mp4v', 'XVID'],
        'h265': ['HEVC', 'H264', 'mp4v'],  
        'mp4v': ['mp4v', 'XVID'],
        'xvid': ['XVID', 'mp4v'],
        'auto': ['H264', 'mp4v', 'XVID', 'MJPG']
    }

    def get_codecs_to_try(self):
        """Get list of codecs to try based on configured codec."""
        # Get codecs for specified format, with fallback
        codecs_to_try = self.CODEC_CONFIGS.get(
            self.codec.lower(), 
            self.CODEC_CONFIGS['auto']  # Better fallback than just ['mp4v']
        )
        
        self.logger.debug(f"Will try codecs in order: {codecs_to_try}")
        return codecs_to_try
    
    def create_video_writer(self, output_path, fps, frame_size):
        codecs_to_try = ['h264_nvenc', 'H264', 'mp4v', 'XVID']  # Prioritize h264_nvenc
        for codec_str in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
                if writer.isOpened():
                    self.writer = writer
                    self.logger.info(f"Successfully initialized codec: {codec_str}")
                    return writer
                else:
                    writer.release()
                    self.logger.error(f"Codec {codec_str} failed to open VideoWriter")
            except Exception as e:
                self.logger.error(f"Codec {codec_str} initialization error: {e}")
        raise RuntimeError(f"No working codec found. Tried: {codecs_to_try}")

    def _initialize_encoder(self):
        try:
            self.create_video_writer(self.output_path, self.fps, self.frame_size)
        except Exception as e:
            self.logger.error(f"Encoder initialization failed: {e}")
            self.release()
            raise
    
    def _initialize_gpu_memory(self):
        try:
            if self.use_cuda:
                self.gpu_frame = cv2.cuda.GpuMat()
                self.logger.info("GPU memory allocated for encoding")
        except Exception as e:
            self.logger.warning(f"GPU memory allocation failed: {e}")
            self.use_cuda = False

    def write_frame(self, frame: np.ndarray) -> bool:
        try:
            if not self.writer or not self.writer.isOpened():
                return False
            if frame is None or frame.size == 0:
                return False
            if frame.shape[:2][::-1] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            if self.use_cuda and self.gpu_frame is not None:
                try:
                    self.gpu_frame.upload(frame)
                    self.writer.write(self.gpu_frame.download())
                except Exception as e:
                    self.logger.warning(f"CUDA write failed, falling back to CPU: {e}")
                    self.use_cuda = False
                    self.writer.write(frame)
            else:
                self.writer.write(frame)
            self.frames_written += 1
            return True
        except Exception as e:
            self.logger.error(f"Error writing frame: {e}")
            return False

    def release(self):
        try:
            if self.writer:
                self.writer.release()
                self.writer = None
            if self.gpu_frame:
                del self.gpu_frame
                self.gpu_frame = None
            self.logger.info(f"Encoder released. Frames written: {self.frames_written}")
        except Exception as e:
            self.logger.error(f"Error releasing encoder: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

# Face Processor
class FaceProcessor:
    def __init__(self, config: DetectionConfig, frame_dims: Tuple[int, int] = (0, 0)):
        self.config = config
        self.frame_dims = frame_dims

    def calculate_deduced_face_region(self, person_bbox: List[float], person_idx: int) -> Optional[Tuple[List[float], float, int]]:
        x1, y1, x2, y2 = map(float, person_bbox)
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
            return None
        w, h = self.frame_dims
        if x2 > w or y2 > h:
            return None
        ph, pw = max(1, y2 - y1), max(1, x2 - x1)
        if ph < 16 or pw < 8:
            return None
        fh = ph * self.config.face_region_ratio
        fy1, fy2 = y1, y1 + fh
        wr = pw * self.config.face_width_reduction
        fx1, fx2 = x1 + wr, x2 - wr
        face_center_x = (x1 + x2) / 2
        face_width = (fx2 - fx1) * 0.7
        fx1 = face_center_x - face_width / 2
        fx2 = face_center_x + face_width / 2
        if fx2 <= fx1:
            fx1, fx2 = x1, x2
        if (fx2 - fx1) < 2:
            cx = (fx1 + fx2) / 2
            fx1, fx2 = cx - 1, cx + 1
        if (fy2 - fy1) < 2:
            fy2 = fy1 + 2
        fx1 = max(0, min(fx1, w))
        fy1 = max(0, min(fy1, h))
        fx2 = max(fx1 + 2, min(fx2, w))
        fy2 = max(fy1 + 2, min(fy2, h))
        if fx2 <= x1 or fy2 <= fy1:
            return None
        confidence = self.config.face_confidence_threshold * 0.5
        return [fx1, fy1, fx2, fy2], confidence, person_idx

    def merge_and_nms_faces(self, detected_faces_tensor: Optional[torch.Tensor], deduced_faces_list: List, device: torch.device, tracks: List) -> List[List[float]]:
        start = time.perf_counter()
        all_faces = []
        track_id_to_bbox = {track.track_id: track.to_ltrb() for track in tracks if track.is_confirmed()}
        if detected_faces_tensor is not None and len(detected_faces_tensor) > 0:
            for face in detected_faces_tensor:
                x1, y1, x2, y2, conf = map(float, face[:5].to(device).tolist())
                if x2 > x1 and y2 > y1 and conf >= self.config.face_confidence_threshold:
                    all_faces.append([x1, y1, x2, y2, conf, 0])
        for face_data in deduced_faces_list:
            if face_data is None:
                continue
            bbox, conf, person_idx = face_data
            x1, y1, x2, y2 = bbox
            if x2 > x1 and y2 > y1 :
                all_faces.append([x1, y1, x2, y2, conf, 1])
        if not all_faces:
            elapsed = (time.perf_counter() - start) * 1000
            logging.info(f"merge_and_nms_faces took {elapsed:.2f} ms")
            return []
        
        merged_faces = custom_nms(all_faces, self.config.face_match_iou_threshold)
        final_faces = []

        for face in merged_faces:
            bbox = face[:4]
            source_type = face[5]
            
            # Find best matching track
            best_track_id = None
            best_iou = 0
            
            for track_id, track_bbox in track_id_to_bbox.items():
                iou = calculate_containment_ratio(bbox, track_bbox)
                #print(f"Track ID: {track_id}, IoU: {iou:.3f}")
                
                if iou > self.config.face_match_iou_threshold and iou > best_iou:
                    best_track_id = track_id
                    best_iou = iou
            
            # Check if face should be skipped due to protected track
            if best_track_id is not None:
                #print(f"Face associated with track ID: {best_track_id}")
                #print(f"no_blur_track_ids: {self.config.no_blur_track_ids}")
                #print(f"Type of best_track_id: {type(best_track_id)}")
                #print(f"Type of no_blur_track_ids items: {[type(x) for x in self.config.no_blur_track_ids]}")
                #print(f"Is {best_track_id} in no_blur_track_ids? {best_track_id in self.config.no_blur_track_ids}")
                
                logging.debug(f"Face associated with track ID {best_track_id}")
                
                if int(best_track_id) in self.config.no_blur_track_ids:
                    #print(f"Skipping face - track {best_track_id} is protected")
                    logging.info(f"Skipping blur for face associated with track ID {best_track_id}")
                    continue  # Skip this face entirely
            
            # Check for duplicate detection (source_type filtering)
            if source_type == 1:
                has_nearby_detected = any(
                    other_face[5] == 0 and calculate_iou(bbox, other_face[:4]) > self.config.face_match_iou_threshold
                    for other_face in merged_faces
                )
                if has_nearby_detected:
                    print("Skipping source_type=1 face due to nearby source_type=0 detection")
                    continue
            
            # Add face to final list (will be blurred)
            final_faces.append(face)
            
        #print(f"Total faces to be processed: {len(final_faces)}")

        if len(final_faces) > self.config.max_detections:
            final_faces = final_faces[:self.config.max_detections]

        face_bboxes = [face[:4] for face in final_faces]
        elapsed = (time.perf_counter() - start) * 1000
        logging.info(f"merge_and_nms_faces took {elapsed:.2f} ms")
        return face_bboxes

# Blur Processor
class BlurProcessor:
    def __init__(self, config: DetectionConfig, use_cuda: bool = True):
        self.config = config
        self.use_cuda = use_cuda and CUDA_AVAILABLE and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.gpu_frame: Optional[cv2.cuda.GpuMat] = None
        self.gaussian_filter = None
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.use_cuda:
            try:
                self.gpu_frame = cv2.cuda.GpuMat()
                self.logger.info("GPU memory allocated for blur processing")
            except Exception as e:
                self.logger.warning(f"CUDA blur init failed: {e}")
                self.use_cuda = False

    def apply_elliptical_blur(self, frame: np.ndarray, detections: List[List[float]], blur_type: str = "object", frame_id: Optional[int] = None, tracks: Optional[List] = None) -> np.ndarray:
        start = time.perf_counter()
        if not detections:
            elapsed = (time.perf_counter() - start) * 1000
            self.logger.debug(f"Frame {frame_id}: apply_elliptical_blur ({blur_type}) took {elapsed:.2f} ms - No detections")
            return frame
        h, w = frame.shape[:2]
        result_frame = frame.copy()
        kernel = self.config.blur_strength if blur_type == "object" else self.config.blur_strength + 2
        if kernel % 2 == 0:
            kernel += 1
        expansion = self.config.blur_expansion * (1.5 if blur_type == "face" else 1.0)
        self.logger.debug(f"Frame {frame_id}: Applying blur with expansion={expansion:.2f}, kernel={kernel} for {blur_type}")
        if self.use_cuda:
            try:
                if self.gaussian_filter is None:
                    self.gaussian_filter = cv2.cuda.createGaussianFilter(
                        cv2.CV_8UC3, cv2.CV_8UC3, (kernel, kernel), 0)
                    self.logger.debug(f"Frame {frame_id}: Created CUDA Gaussian filter with kernel {kernel}")
            except Exception as e:
                self.logger.error(f"Frame {frame_id}: Failed to create CUDA Gaussian filter: {e}")
                self.use_cuda = False
                self.gaussian_filter = None
        track_id_to_bbox = {track.track_id: track.to_ltrb() for track in tracks if track.is_confirmed()} if tracks else {}
        for idx, bbox in enumerate(detections):
            try:
                x1, y1, x2, y2 = map(int, bbox[:4])
                self.logger.debug(f"Frame {frame_id}: Processing bbox {idx}: [{x1}, {y1}, {x2}, {y2}]")
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    self.logger.warning(f"Frame {frame_id}: Invalid bbox {idx} for {blur_type}")
                    continue
                associated_track_id = None
                for track_id, track_bbox in track_id_to_bbox.items():
                    if calculate_iou(bbox, track_bbox) > self.config.face_match_iou_threshold:
                        associated_track_id = track_id
                        break
                if associated_track_id is not None and associated_track_id in self.config.no_blur_track_ids:
                    self.logger.debug(f"Frame {frame_id}: Skipping blur for bbox {idx} (track ID {associated_track_id} in no_blur_track_ids)")
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                ell_w = max(self.config.min_ellipse_size, int((x2 - x1) * expansion / 2))
                ell_h = max(self.config.min_ellipse_size, int((y2 - y1) * expansion / 2))
                margin = max(2, kernel // 2)
                rx1 = max(0, cx - ell_w - margin)
                ry1 = max(0, cy - ell_h - margin)
                rx2 = min(cx + ell_w + margin, w)
                ry2 = min(cy + ell_h + margin, h)
                if rx2 <= rx1 or ry2 <= ry1:
                    self.logger.warning(f"Frame {frame_id}: Invalid ROI for bbox {idx}")
                    continue
                roi = result_frame[ry1:ry2, rx1:rx2].copy()
                if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                    self.logger.warning(f"Frame {frame_id}: Empty ROI for bbox {idx}")
                    continue
                roi_blurred = self._apply_blur_to_roi(roi, kernel, frame_id, idx)
                if roi_blurred is None:
                    continue
                mask = np.zeros((ry2-ry1, rx2-rx1), dtype=np.uint8)
                lcx, lcy = cx - rx1, cy - ry1
                if 0 <= lcx < mask.shape[1] and 0 <= lcy < mask.shape[0] and ell_w > 0 and ell_h > 0:
                    cv2.ellipse(mask, (lcx, lcy), (ell_w, ell_h), 0, 0, 360, 255, -1)
                    self.logger.debug(f"Frame {frame_id}: Ellipse mask created for bbox {idx}")
                    if np.sum(mask) == 0:
                        self.logger.warning(f"Frame {frame_id}: Empty mask for bbox {idx}")
                        continue
                    smooth_kernel = min(15, max(3, min(ell_w, ell_h) // 4))
                    if smooth_kernel % 2 == 0:
                        smooth_kernel += 1
                    smooth_mask = cv2.GaussianBlur(mask, (smooth_kernel, smooth_kernel), 0).astype(np.float32) / 255.0
                    smooth_mask_3d = smooth_mask[..., np.newaxis]
                    roi_result = (smooth_mask_3d * roi_blurred + (1.0 - smooth_mask_3d) * roi).astype(np.uint8)
                    result_frame[ry1:ry2, rx1:rx2] = roi_result
                    self.logger.debug(f"Frame {frame_id}: Blurred ROI blended for bbox {idx}")
                else:
                    self.logger.warning(f"Frame {frame_id}: Invalid ellipse parameters for bbox {idx}")
            except Exception as e:
                self.logger.error(f"Frame {frame_id}: Blur failed for {blur_type} bbox {idx}: {e}")
                continue
        elapsed = (time.perf_counter() - start) * 1000
        self.logger.info(f"Frame {frame_id}: apply_elliptical_blur ({blur_type}) took {elapsed:.2f} ms")
        return result_frame

    def _apply_blur_to_roi(self, roi: np.ndarray, kernel: int, frame_id: Optional[int], bbox_idx: int) -> Optional[np.ndarray]:
        try:
            if self.use_cuda and self.gaussian_filter and roi.shape[0] >= 100 and roi.shape[1] >= 100:
                try:
                    if not roi.flags['C_CONTIGUOUS']:
                        roi = np.ascontiguousarray(roi)
                    gpu_roi = cv2.cuda.GpuMat(roi.shape[0], roi.shape[1], cv2.CV_8UC3)
                    gpu_roi.upload(roi)
                    gpu_blurred = cv2.cuda.GpuMat(roi.shape[0], roi.shape[1], cv2.CV_8UC3)
                    self.gaussian_filter.apply(gpu_roi, gpu_blurred)
                    roi_blurred = gpu_blurred.download()
                    self.logger.debug(f"Frame {frame_id}: CUDA blur applied successfully for bbox {bbox_idx}")
                    return roi_blurred
                except Exception as e:
                    self.logger.warning(f"Frame {frame_id}: CUDA blur failed for bbox {bbox_idx}: {e}")
                    self.use_cuda = False
                    self.gaussian_filter = None
            roi_blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
            self.logger.debug(f"Frame {frame_id}: CPU blur applied for bbox {bbox_idx}")
            return roi_blurred
        except Exception as e:
            self.logger.error(f"Frame {frame_id}: All blur methods failed for bbox {bbox_idx}: {e}")
            return None

    def __del__(self):
        if self.gpu_frame:
            del self.gpu_frame
            self.gpu_frame = None
        if self.gaussian_filter:
            del self.gaussian_filter
            self.gaussian_filter = None

# Main Application
class YOLOv9DeepSortApp:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = select_device(self.config.device)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.performance_monitor = PerformanceMonitor()
        self.object_model: Optional[DetectMultiBackend] = None
        self.face_model: Optional[DetectMultiBackend] = None
        self.face_processor: Optional[FaceProcessor] = None
        self.blur_processor: Optional[BlurProcessor] = None
        self.object_tracker: Optional[DeepSort] = None
        self.frame_buffer = AsyncFrameBuffer(self.config.buffer_size) if self.config.enable_async_processing else None
        self._lock = threading.Lock()
        self.last_detections = {
            'object_bboxes': [], 'object_classes': [], 'object_confidences': [],
            'merged_face_bboxes': [], 'tracks': []
        }
        self._initialize_models()
        self._initialize_tracker()

    def _initialize_models(self):
        try:
            self.logger.info("Loading models...")
            self.object_model = DetectMultiBackend(self.config.object_model_path, device=self.device)
            self.face_model = DetectMultiBackend(self.config.face_model_path, device=self.device)
            self.face_processor = FaceProcessor(self.config)
            self.blur_processor = BlurProcessor(self.config, use_cuda=self.config.enable_cuda_decoder)
            dummy_input = torch.zeros((1, 3, self.config.input_size, self.config.input_size)).to(self.device)
            with torch.no_grad():
                self.object_model(dummy_input)
                self.face_model(dummy_input)
            self.logger.info("Models loaded and warmed up successfully")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise

    def _initialize_tracker(self):
        try:
            self.object_tracker = DeepSort(
                max_age=self.config.deepsort_max_age,
                n_init=self.config.deepsort_min_hits,
                max_iou_distance=self.config.deepsort_iou_tracker,
                embedder="mobilenet"
            )
            self.logger.info("DeepSORT tracker initialized")
        except Exception as e:
            self.logger.error(f"Tracker initialization failed: {e}")
            raise

    def process_frame(self, frame: np.ndarray, decoder: OptimizedCUDADecoder, frame_idx: int) -> Tuple[List[List[float]], List[int], List[float], List[List[float]]]:
        start = time.perf_counter()
        width, height = frame.shape[1], frame.shape[0]
        self.face_processor.frame_dims = (width, height)
        img_letterboxed, _, _ = decoder.letterbox_gpu(frame, (self.config.input_size, self.config.input_size))
        im = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        im = np.ascontiguousarray(im)
        im_tensor = torch.from_numpy(im).to(self.device)
        im_tensor = im_tensor.half() if self.object_model.fp16 else im_tensor.float()
        im_tensor /= 255.0
        im_tensor = im_tensor.unsqueeze(0)
        try:
            with torch.no_grad():
                pred_obj = self.object_model(im_tensor)
                det_obj = non_max_suppression(
                    pred_obj[0] if isinstance(pred_obj, (list, tuple)) else pred_obj,
                    self.config.object_confidence_threshold,
                    self.config.iou_threshold,
                    classes=self.config.classes,
                    max_det=self.config.max_detections
                )
                pred_face = self.face_model(im_tensor)
                det_face = non_max_suppression(
                    pred_face[0] if isinstance(pred_face, (list, tuple)) else pred_face,
                    self.config.face_confidence_threshold,
                    self.config.iou_threshold,
                    max_det=self.config.max_detections
                )
        except Exception as e:
            self.logger.error(f"Frame {frame_idx}: Inference failed: {e}")
            return [], [], [], []
        object_bboxes = []
        object_classes = []
        object_confidences = []
        if det_obj and len(det_obj) > 0 and len(det_obj[0]) > 0:
            det0 = det_obj[0]
            det0[:, :4] = scale_boxes(img_letterboxed.shape[:2], det0[:, :4], frame.shape[:2]).round()
            for i in range(len(det0)):
                x1, y1, x2, y2 = det0[i, :4].cpu().numpy()
                cls = int(det0[i, 5].cpu().numpy())
                conf = float(det0[i, 4].cpu().numpy())
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
                    object_bboxes.append([x1, y1, x2, y2])
                    object_classes.append(cls)
                    object_confidences.append(conf)
        detected_faces_tensor = det_face[0] if det_face and len(det_face) > 0 and len(det_face[0]) > 0 else None
        if detected_faces_tensor is not None:
            detected_faces_tensor[:, :4] = scale_boxes(img_letterboxed.shape[:2], detected_faces_tensor[:, :4], frame.shape[:2]).round()
        deduced_faces = []
        for idx, (bbox, cls) in enumerate(zip(object_bboxes, object_classes)):
            if cls == 0:
                face_data = self.face_processor.calculate_deduced_face_region(bbox, idx)
                if face_data:
                    deduced_faces.append(face_data)
        tracks = self.track_objects(frame, object_bboxes, object_classes, object_confidences, frame_idx)
        merged_face_bboxes = self.face_processor.merge_and_nms_faces(detected_faces_tensor, deduced_faces, self.device, tracks)
        
        if len(object_bboxes) != len(object_classes) or len(object_bboxes) != len(object_confidences):
            self.logger.error(f"Frame {frame_idx}: Detection mismatch")
            min_len = min(len(object_bboxes), len(object_classes), len(object_confidences))
            object_bboxes = object_bboxes[:min_len]
            object_classes = object_classes[:min_len]
            object_confidences = object_confidences[:min_len]
        self.last_detections['tracks'] = tracks
        elapsed = (time.perf_counter() - start) * 1000
        self.performance_monitor.record_inference_time(elapsed)
        self.logger.info(f"Frame {frame_idx}: process_frame took {elapsed:.2f} ms")
        return object_bboxes, object_classes, object_confidences, merged_face_bboxes

    def track_objects(self, frame: np.ndarray, object_bboxes: List[List[float]], object_classes: List[int], object_confidences: List[float], frame_idx: int) -> List:
        start = time.perf_counter()
        detections = []
        for i, (bbox, cls, conf) in enumerate(zip(object_bboxes, object_classes, object_confidences)):
            x1, y1, x2, y2 = map(float, bbox)
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                detections.append(([x1, y1, w, h], conf, cls))
            else:
                self.logger.debug(f"Frame {frame_idx}: Invalid bbox {bbox} for tracking")
        try:
            tracks = self.object_tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            self.logger.error(f"Frame {frame_idx}: Tracking failed: {e}")
            tracks = []
        elapsed = (time.perf_counter() - start) * 1000
        self.logger.info(f"Frame {frame_idx}: track_objects took {elapsed:.2f} ms")
        return tracks

    def get_color(self,track_id: int) -> Tuple[int, int, int]:
        """
        Generate a BGR color for a given track ID using a hash.

        Args:
            track_id (int): Unique identifier for the track.

        Returns:
            Tuple[int, int, int]: BGR color tuple (blue, green, red).
        """
        # Use hash to generate pseudo-random but consistent colors
        h = hash(str(track_id)) & 0xFFFFFF
        r = (h & 0xFF0000) >> 16
        g = (h & 0x00FF00) >> 8
        b = h & 0x0000FF
        return (b, g, r)
    
    def validate_bbox(self, x1: float, y1: float, x2: float, y2: float, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Ensure bounding box coordinates are within frame boundaries.

        Args:
            x1 (float): Left x-coordinate.
            y1 (float): Top y-coordinate.
            x2 (float): Right x-coordinate.
            y2 (float): Bottom y-coordinate.
            frame_shape (Tuple[int, int]): Frame dimensions (height, width).

        Returns:
            Tuple[int, int, int, int]: Validated (x1, y1, x2, y2) coordinates.

        Raises:
            ValueError: If frame_shape is invalid.
        """
        if not isinstance(frame_shape, (tuple, list)) or len(frame_shape) < 2:
            raise ValueError("frame_shape must be a tuple/list with at least (height, width)")
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Frame dimensions must be positive")

        # Convert to int and clamp coordinates
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))
        return x1, y1, x2, y2

    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], color: Tuple[int, int, int], thickness: int = 4) -> None:
        """
        Draw a rectangle on the frame with validated coordinates.

        Args:
            frame (np.ndarray): Input frame (BGR image).
            bbox (Tuple[float, float, float, float]): Bounding box (x1, y1, x2, y2).
            color (Tuple[int, int, int]): BGR color for the rectangle.
            thickness (int, optional): Line thickness. Defaults to 2.

        Raises:
            ValueError: If frame or color is invalid.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be a 3-channel BGR image")
        if not isinstance(color, (tuple, list)) or len(color) != 3 or not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            raise ValueError("Color must be a 3-tuple of integers in [0, 255]")
        if thickness < 0:
            raise ValueError("Thickness must be non-negative")

        x1, y1, x2, y2 = self.validate_bbox(*bbox, frame.shape)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> bool:
            """Check if two rectangles overlap."""
            x1_1, y1_1, x2_1, y2_1 = rect1
            x1_2, y1_2, x2_2, y2_2 = rect2
            
            # No overlap if one rectangle is to the left, right, above, or below the other
            return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)
    
    def draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int],
        font_scale: float,
        font_thickness: int,
        bg_alpha: float,
        text_regions: List[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Draw text with a semi-transparent background, avoiding overlaps.
        
        Args:
            frame: Input frame (BGR image)
            text: Text to draw
            pos: BBox top-left position (x, y)
            color: BGR color for background
            font_scale: Font scale for text
            font_thickness: Font thickness
            bg_alpha: Background transparency (0-1)
            text_regions: Existing text regions to avoid
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Updated list of text regions
        """
        if not 0 <= bg_alpha <= 1:
            raise ValueError("bg_alpha must be between 0 and 1")
        
        # Get text dimensions 
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        baseline = 2  # Small default baseline
        
        # Define padding around text
        padding = 4
        
        # Calculate background rectangle dimensions
        bg_w = text_w + 2 * padding
        bg_h = text_h + baseline + 2 * padding
        
        bbox_x, bbox_y = pos
        frame_h, frame_w = frame_shape[:2]
        
        # Try different positions in priority order
        positions_to_try = [
            # Above bbox  
            (bbox_x, bbox_y - bg_h - 5),
            # Right of bbox
            (bbox_x + 40, bbox_y),
            # Left of bbox
            (bbox_x - bg_w - 10, bbox_y),
            # Below bbox
            (bbox_x, bbox_y + 40)
        ]
        
        best_position = None
        
        for bg_x, bg_y in positions_to_try:
            # Clamp background rectangle to frame boundaries
            bg_x = max(0, min(bg_x, frame_w - bg_w))
            bg_y = max(0, min(bg_y, frame_h - bg_h))
            
            # Define the background rectangle
            bg_rect = (bg_x, bg_y, bg_x + bg_w, bg_y + bg_h)
            
            # Check for overlaps with existing text regions
            overlaps = any(self._rectangles_overlap(bg_rect, existing) for existing in text_regions)
            
            if not overlaps:
                best_position = (bg_x, bg_y)
                break
        
        # If no non-overlapping position found, try with vertical offsets
        if best_position is None:
            bg_x, bg_y = positions_to_try[0]  # Start with below-bbox position
            bg_x = max(0, min(bg_x, frame_w - bg_w))
            
            # Try shifting down in small increments
            for offset in range(0, min(50, frame_h - bg_y - bg_h), bg_h // 2):
                test_y = bg_y + offset
                if test_y + bg_h > frame_h:
                    break
                    
                bg_rect = (bg_x, test_y, bg_x + bg_w, test_y + bg_h)
                overlaps = any(self._rectangles_overlap(bg_rect, existing) for existing in text_regions)
                
                if not overlaps:
                    best_position = (bg_x, test_y)
                    break
        
        # Use first position if still no good spot found
        if best_position is None:
            bg_x, bg_y = positions_to_try[0]
            bg_x = max(0, min(bg_x, frame_w - bg_w))
            bg_y = max(0, min(bg_y, frame_h - bg_h))
            best_position = (bg_x, bg_y)
        
        # Final background rectangle coordinates
        bg_x, bg_y = best_position
        bg_x1, bg_y1 = bg_x, bg_y
        bg_x2, bg_y2 = bg_x + bg_w, bg_y + bg_h
        
        # Calculate text position within the background rectangle
        # Text baseline should be at bottom of rectangle minus padding
        text_x = bg_x1 + padding
        text_y = bg_y2 - padding - baseline
        
        # Ensure we don't draw outside frame bounds
        if (bg_x2 <= frame_w and bg_y2 <= frame_h and 
            bg_x1 >= 0 and bg_y1 >= 0 and bg_x1 < bg_x2 and bg_y1 < bg_y2):
            
            # Draw semi-transparent background
            roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
            if roi.size > 0:
                # Create colored overlay for the ROI
                overlay = np.full_like(roi, color, dtype=np.uint8)
                
                # Blend with original
                blended = cv2.addWeighted(overlay, bg_alpha, roi, 1.0 - bg_alpha, 0.0)
                frame[bg_y1:bg_y2, bg_x1:bg_x2] = blended
                
                # Draw thin white border for contrast
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)
                
                # Draw text with black outline for readability
                cv2.putText(frame, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                           (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                
                # Draw main white text
                cv2.putText(frame, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                           (0, 0, 0), font_thickness, cv2.LINE_AA)
                
                # Add this region to the list
                text_regions.append((bg_x1, bg_y1, bg_x2, bg_y2))
        
        return text_regions

    def visualize_tracks(self,frame, object_bboxes, object_classes, tracks, config, frame_idx, logger):
        """Visualize object and track bounding boxes with track IDs."""
        result_frame = frame.copy()
        text_regions = []

        # Draw object bounding boxes
        #for i, bbox in enumerate(object_bboxes):
        #    if i >= len(object_classes):
        #        continue
        #    self.draw_bbox(result_frame, bbox, (0, 255, 0))

        # Draw track bounding boxes and IDs
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            track_id = track.track_id
            color = self.get_color(track_id)
            self.draw_bbox(result_frame, ltrb, color)
            text = f"#{track_id}"
            x1, y1, _, _ = self.validate_bbox(*ltrb, result_frame.shape)
            text_regions = self.draw_text_with_background(
                result_frame, text, (x1 + 2, y1 - 5), color,
                config.track_id_font_scale, config.track_id_font_thickness,
                config.track_id_bg_alpha, text_regions, result_frame.shape
            )
            logger.debug(f"Frame {frame_idx}: Drew track ID {track_id} at {x1 + 2},{y1 - 5}")

        return result_frame
    
    def draw(self, frame: np.ndarray, tracks: List, face_bboxes: List[List[float]], object_bboxes: List[List[float]], object_classes: List[int], object_conf: List[float], frame_idx: int) -> np.ndarray:
        start = time.perf_counter()
                
        result_frame = self.visualize_tracks(frame, object_bboxes, object_classes, tracks, config, frame_idx, self.logger)
        
        for face in face_bboxes:
            x1, y1, x2, y2 = map(int, face)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elapsed = (time.perf_counter() - start) * 1000
        self.logger.info(f"Frame {frame_idx}: draw took {elapsed:.2f} ms")
        return result_frame

    def _add_performance_overlay(self, frame: np.ndarray, frame_idx: int, decoder: OptimizedCUDADecoder) -> np.ndarray:
        stats = self.performance_monitor.get_stats()
        fps = stats.get('fps', 0)
        memory = stats.get('memory_usage', {}).get('avg', 0)
        gpu_memory = stats.get('gpu_memory', {}).get('avg', 0) if stats.get('gpu_memory') else 0
        progress = decoder.get_progress()
        texts = [
            f"FPS: {fps:.1f}",
            f"Progress: {progress:.1f}%",
            f"Frame: {frame_idx}",
            f"Skip Frames: {self.config.skip_frames}",
            f"Blur Expansion: {self.config.blur_expansion:.2f}",
            f"Memory: {memory:.1f} MB",
            f"GPU Memory: {gpu_memory:.1f} MB"
        ]
        max_width = 0
        text_height = 0
        for text in texts:
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            max_width = max(max_width, w)
            text_height = max(text_height, h)
        overlay = frame.copy()
        bg_x1, bg_y1 = 10, 10
        bg_x2 = bg_x1 + max_width + 20
        bg_y2 = bg_y1 + (len(texts) * (text_height + 20)) + 10
        if bg_x2 > frame.shape[1]:
            bg_x2 = frame.shape[1]
        if bg_y2 > frame.shape[0]:
            bg_y2 = frame.shape[0]
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0.0, frame)
        for i, text in enumerate(texts):
            cv2.putText(
                frame, text, (bg_x1 + 10, bg_y1 + (i + 1) * (text_height + 20)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

    def _process_video_sync(self, decoder: OptimizedCUDADecoder, encoder: Optional[OptimizedCUDAEncoder], display: bool):
        frame_idx = 0
        win_name = 'YOLOv9 DeepSort Blur Detection + DeepSort Tracking'
        if display:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            width, height = decoder.get_frame_size()
            aspect_ratio = width / height
            max_width = 1280
            max_height = int(max_width / aspect_ratio)
            if max_height > 720:
                max_height = 720
                max_width = int(max_height * aspect_ratio)
            cv2.resizeWindow(win_name, max_width, max_height)
        try:
            while True:
                start_time = time.perf_counter()
                ret, frame = decoder.read_and_preprocess()
                if not ret:
                    self.logger.info("End of video reached")
                    break
                frame_idx += 1
                with self._lock:
                    if frame_idx % max(1, self.config.skip_frames) == 0:
                        object_bboxes, object_classes, object_conf, merged_faces = self.process_frame(frame, decoder, frame_idx)
                        tracks = self.track_objects(frame, object_bboxes, object_classes, object_conf, frame_idx)
                        self.last_detections = {
                            'object_bboxes': object_bboxes,
                            'object_classes': object_classes,
                            'object_confidences': object_conf,
                            'merged_face_bboxes': merged_faces,
                            'tracks': tracks
                        }
                    else:
                        self.logger.debug(f"Frame {frame_idx}: Using cached detections")
                        try:
                            self.object_tracker.update_tracks([], frame=frame)
                            self.logger.debug(f"Frame {frame_idx}: Updated tracker with empty detections")
                        except Exception as e:
                            self.logger.error(f"Frame {frame_idx}: Tracker update failed: {e}")
                processed_frame = frame.copy()
                if self.config.blur_faces and self.last_detections['merged_face_bboxes']:
                    processed_frame = self.blur_processor.apply_elliptical_blur(
                        processed_frame, self.last_detections['merged_face_bboxes'], "face", frame_idx, self.last_detections['tracks'])
                    self.performance_monitor.record_blur_time((time.perf_counter() - start_time) * 1000)
                if self.config.blur_objects and self.last_detections['object_bboxes']:
                    processed_frame = self.blur_processor.apply_elliptical_blur(
                        processed_frame, self.last_detections['object_bboxes'], "object", frame_idx, self.last_detections['tracks'])
                    self.performance_monitor.record_blur_time((time.perf_counter() - start_time) * 1000)
                if self.config.draw_detections:
                    processed_frame = self.draw(
                        processed_frame, self.last_detections['tracks'], 
                        self.last_detections['merged_face_bboxes'],
                        self.last_detections['object_bboxes'], 
                        self.last_detections['object_classes'],
                        self.last_detections['object_confidences'], 
                        frame_idx)
                processed_frame = self._add_performance_overlay(processed_frame, frame_idx, decoder)
                if encoder:
                    encoder.write_frame(processed_frame)
                if display:
                    cv2.imshow(win_name, processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)
                    elif key == ord('r') and not encoder and self.config.output_path and Path(self.config.output_path).suffix.lower() in {'.mp4', '.avi'}:
                        try:
                            encoder = OptimizedCUDAEncoder(
                                self.config.output_path, decoder.fps, decoder.get_frame_size(),
                                self.config.enable_cuda_encoder, self.config.output_codec, self.config.output_bitrate)
                            self.logger.info(f"Started recording to: {self.config.output_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to start recording: {e}")
                    elif key == ord('s') and encoder:
                        encoder.release()
                        encoder = None
                        self.logger.info("Recording stopped")
                frame_time = (time.perf_counter() - start_time) * 1000
                self.performance_monitor.record_frame_time(frame_time)
                self.performance_monitor.record_memory_usage()
                if frame_idx % 50 == 0 or frame_idx == decoder.total_frames:
                    stats = self.performance_monitor.get_stats()
                    self.logger.info(f"Frame {frame_idx}: {stats.get('fps', 0):.1f} FPS, "
                                    f"Memory: {stats.get('memory_usage', {}).get('avg', 0):.1f} MB")
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            if display:
                cv2.waitKey(0)
            raise
        finally:
            if encoder:
                encoder.release()
            if display:
                cv2.destroyAllWindows()

    def _process_video_async(self, decoder: OptimizedCUDADecoder, encoder: Optional[OptimizedCUDAEncoder], display: bool):
        def frame_reader():
            try:
                while not self.frame_buffer.is_stopped():
                    ret, frame = decoder.read_and_preprocess()
                    if not ret:
                        self.frame_buffer.stop()
                        break
                    if not self.frame_buffer.put((frame, decoder.frame_count)):
                        self.logger.warning(f"Frame {decoder.frame_count}: Buffer full, dropping frame")
            except Exception as e:
                self.logger.error(f"Frame reader error: {e}")
                self.frame_buffer.stop()

        reader_thread = threading.Thread(target=frame_reader, daemon=True)
        reader_thread.start()
        frame_idx = 0
        win_name = 'YOLOv9 DeepSort Blur Detection + DeepSort Tracking'
        if display:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            width, height = decoder.get_frame_size()
            aspect_ratio = width / height
            max_width = 1280
            max_height = int(max_width / aspect_ratio)
            if max_height > 720:
                max_height = 720
                max_width = int(max_height * aspect_ratio)
            cv2.resizeWindow(win_name, max_width, max_height)
        try:
            while not self.frame_buffer.is_stopped():
                start_time = time.perf_counter()
                item = self.frame_buffer.get()
                if item is None:
                    break
                frame, frame_idx = item
                with self._lock:
                    if frame_idx % max(1, self.config.skip_frames) == 0:
                        object_bboxes, object_classes, object_conf, merged_faces = self.process_frame(frame, decoder, frame_idx)
                        tracks = self.track_objects(frame, object_bboxes, object_classes, object_conf, frame_idx)
                        self.last_detections = {
                            'object_bboxes': object_bboxes,
                            'object_classes': object_classes,
                            'object_confidences': object_conf,
                            'merged_face_bboxes': merged_faces,
                            'tracks': tracks
                        }
                    else:
                        self.logger.debug(f"Frame {frame_idx}: Using cached detections")
                        try:
                            self.object_tracker.update_tracks([], frame=frame)
                            self.logger.debug(f"Frame {frame_idx}: Updated tracker with empty detections")
                        except Exception as e:
                            self.logger.error(f"Frame {frame_idx}: Tracker update failed: {e}")
                processed_frame = frame.copy()
                if self.config.blur_faces and self.last_detections['merged_face_bboxes']:
                    processed_frame = self.blur_processor.apply_elliptical_blur(
                        processed_frame, self.last_detections['merged_face_bboxes'], "face", frame_idx, self.last_detections['tracks'])
                    self.performance_monitor.record_blur_time((time.perf_counter() - start_time) * 1000)
                if self.config.blur_objects and self.last_detections['object_bboxes']:
                    processed_frame = self.blur_processor.apply_elliptical_blur(
                        processed_frame, self.last_detections['object_bboxes'], "object", frame_idx, self.last_detections['tracks'])
                    self.performance_monitor.record_blur_time((time.perf_counter() - start_time) * 1000)
                if self.config.draw_detections:
                    processed_frame = self.draw(
                        processed_frame, self.last_detections['tracks'], 
                        self.last_detections['merged_face_bboxes'],
                        self.last_detections['object_bboxes'], 
                        self.last_detections['object_classes'],
                        self.last_detections['object_confidences'], 
                        frame_idx)
                processed_frame = self._add_performance_overlay(processed_frame, frame_idx, decoder)
                if encoder:
                    encoder.write_frame(processed_frame)
                if display:
                    cv2.imshow(win_name, processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.frame_buffer.stop()
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)
                    elif key == ord('r') and not encoder and self.config.output_path and Path(self.config.output_path).suffix.lower() in {'.mp4', '.avi'}:
                        try:
                            encoder = OptimizedCUDAEncoder(
                                self.config.output_path, decoder.fps, decoder.get_frame_size(),
                                self.config.enable_cuda_encoder, self.config.output_codec, self.config.output_bitrate)
                            self.logger.info(f"Started recording to: {self.config.output_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to start recording: {e}")
                    elif key == ord('s') and encoder:
                        encoder.release()
                        encoder = None
                        self.logger.info("Recording stopped")
                frame_time = (time.perf_counter() - start_time) * 1000
                self.performance_monitor.record_frame_time(frame_time)
                self.performance_monitor.record_memory_usage()
                if frame_idx % 50 == 0:
                    stats = self.performance_monitor.get_stats()
                    self.logger.info(f"Frame {frame_idx}: {stats.get('fps', 0):.1f} FPS, "
                                    f"Memory: {stats.get('memory_usage', {}).get('avg', 0):.1f} MB")
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during async processing: {e}")
            if display:
                cv2.waitKey(0)
            raise
        finally:
            self.frame_buffer.stop()
            reader_thread.join()
            if encoder:
                encoder.release()
            if display:
                cv2.destroyAllWindows()

    def run_yolov9_deepsort(self, video_path: str, output_path: Optional[str] = None, display: bool = True):
        self.logger.info(f"Starting video processing: {video_path}")
        try:
            with OptimizedCUDADecoder(video_path, self.config.enable_cuda_decoder) as decoder:
                encoder = None
                if output_path:
                    encoder = OptimizedCUDAEncoder(
                        output_path, decoder.fps, decoder.get_frame_size(),
                        self.config.enable_cuda_encoder, self.config.output_codec, self.config.output_bitrate)
                    self.logger.info(f"Output video: {output_path}")
                if self.config.enable_async_processing:
                    self._process_video_async(decoder, encoder, display)
                else:
                    self._process_video_sync(decoder, encoder, display)
        except Exception as e:
            self.logger.error(f"Video processing error: {e}")
            if display:
                cv2.waitKey(0)
            raise
        finally:
            if display:
                cv2.destroyAllWindows()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("Processing completed")
    
    def run(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        yield_frame: bool = False
    ) -> Optional[Generator[np.ndarray, None, None]]:
        """
        Process a video using YOLOv9 for object detection and DeepSORT for tracking.

        Args:
            video_path (str): Path to the input video file.
            output_path (Optional[str]): Path to save the processed video. Defaults to None.
            display (bool): Whether to display the video during processing. Defaults to True.
            yield_frame (bool): Whether to yield processed frames for streaming (e.g., Gradio). Defaults to False.

        Returns:
            Optional[Generator[np.ndarray, None, None]]: Yields processed frames as numpy arrays if yield_frame is True.
            None if yield_frame is False.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If output_path directory is invalid or config is misconfigured.
            RuntimeError: If video processing fails.
        """
        self.logger.info(f"Starting video processing: {video_path}")

        # Validate input video path
        import os
        if not os.path.isfile(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if output_path and not os.path.isdir(os.path.dirname(output_path) or '.'):
            self.logger.error(f"Output directory does not exist: {os.path.dirname(output_path)}")
            raise ValueError(f"Output directory does not exist: {os.path.dirname(output_path)}")

        try:
            with OptimizedCUDADecoder(video_path, self.config.enable_cuda_decoder) as decoder:
                encoder = None
                if output_path:
                    encoder = OptimizedCUDAEncoder(
                        output_path,
                        decoder.fps,
                        decoder.get_frame_size(),
                        self.config.enable_cuda_encoder,
                        self.config.output_codec,
                        self.config.output_bitrate
                    )
                    self.logger.info(f"Output video: {output_path}")

                # Choose processing mode
                if self.config.enable_async_processing:
                    frame_generator = self._process_video_async(decoder, encoder, display)
                else:
                    frame_generator = self._process_video_sync(decoder, encoder, display)

                # Handle frame processing and yielding
                if yield_frame:
                    for frame in frame_generator:
                        if frame is not None:
                            # Convert frame to RGB for Gradio (OpenCV uses BGR)
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            yield frame_rgb
                else:
                    # Consume generator without yielding (for non-streaming case)
                    for frame in frame_generator:
                        if display and frame is not None:
                            cv2.imshow('YOLOv9 + DeepSORT', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

        except (cv2.error, RuntimeError, IOError) as e:
            self.logger.error(f"Video processing error: {e}")
            if display:
                cv2.waitKey(1000)  # Wait briefly to show error state
            raise
        finally:
            if display:
                cv2.destroyAllWindows()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("Processing completed")

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv9 Detection + DeepSORT Tracking with CUDA Optimization")
    parser.add_argument('--video', required=True, help="Input video path")
    parser.add_argument('--output', default=None, help="Output video path (mp4 or avi)")
    parser.add_argument('--object-model', default="yolov9-m.pt", help="YOLOv9 object model path")
    parser.add_argument('--face-model', default="yolov9-m.pt", help="YOLOv9 face model path")
    parser.add_argument('--conf', type=float, default=0.25, help="Object detection confidence threshold")
    parser.add_argument('--face-conf', type=float, default=0.25, help="Face detection confidence threshold")
    parser.add_argument('--iou', type=float, default=0.45, help="IoU threshold for object NMS")
    parser.add_argument('--face-iou', type=float, default=0.15, help="IoU threshold for face NMS merging")
    parser.add_argument('--draw', action='store_true', help="Draw detection boxes")
    parser.add_argument('--blur-objects', action='store_true', default=False, help="Blur detected objects")
    parser.add_argument('--blur-faces', action='store_true', default=True, help="Blur detected faces")
    parser.add_argument('--blur-strength', type=int, default=21, help="Blur kernel size")
    parser.add_argument('--blur-expansion', type=float, default=2.0, help="Blur ellipse expansion factor")
    parser.add_argument('--face-region-ratio', type=float, default=0.25, help="Face region ratio from person height")
    parser.add_argument('--face-width-reduction', type=float, default=0.15, help="Face width reduction ratio")
    parser.add_argument('--min-ellipse-size', type=int, default=5, help="Minimum ellipse size for blurring")
    parser.add_argument('--max-detections', type=int, default=200, help="Maximum detections per frame")
    parser.add_argument('--input-size', type=int, default=640, help="Model input size")
    parser.add_argument('--device', default='0', help="CUDA device or 'cpu'")
    parser.add_argument('--max-age', type=int, default=15, help="DeepSORT: max age")
    parser.add_argument('--min-hits', type=int, default=2, help="DeepSORT: min hits")
    parser.add_argument('--iou-tracker', type=float, default=0.3, help="DeepSORT: IOU threshold")
    parser.add_argument('--disable-cuda-decoder', action='store_true', help="Disable CUDA video decoder")
    parser.add_argument('--disable-cuda-encoder', action='store_true', help="Disable CUDA video encoder")
    parser.add_argument('--output-codec', type=str, default='h264', 
                        choices=['h264', 'h265', 'x264', 'x265', 'xvid', 'mjpeg'], 
                        help='Output video codec (x264 is alias for h264, x265 for h265)')
    parser.add_argument('--output-bitrate', type=int, default=5000000, help='Output video bitrate (bits/sec)')
    parser.add_argument('--skip-frames', type=int, default=1, help="Process every nth frame (1 = process all frames)")
    parser.add_argument('--async-processing', action='store_true', help="Enable asynchronous frame processing")
    parser.add_argument('--no-display', action='store_true', help="Disable video display")
    parser.add_argument('--classes', type=int, nargs='+', default=[0],#, 2, 3, 5, 7], 
                        help="List of class IDs to detect (default: 0=person, 2=car, 3=motorbike, 5=bus, 7=truck)")
    parser.add_argument('--no-blur-track-ids', type=int, nargs='+', default=[],
                        help="List of track IDs for which associated faces should not be blurred")
    parser.add_argument('--track-id-font-scale', type=float, default=1.0, help="Font scale for track ID text")
    parser.add_argument('--track-id-font-thickness', type=int, default=2, help="Font thickness for track ID text")
    parser.add_argument('--track-id-bg-alpha', type=float, default=0.85, help="Background opacity for track ID text")
    parser.add_argument('--log-file', default=None, help="Log file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_logging(level="INFO", log_file=args.log_file)
    config = DetectionConfig(
        object_model_path=args.object_model,
        face_model_path=args.face_model,
        output_path=args.output,
        object_confidence_threshold=args.conf,
        face_confidence_threshold=args.face_conf,
        iou_threshold=args.iou,
        face_match_iou_threshold=args.face_iou,
        face_region_ratio=args.face_region_ratio,
        face_width_reduction=args.face_width_reduction,
        draw_detections=args.draw,
        blur_objects=args.blur_objects,
        blur_faces=args.blur_faces,
        blur_strength=args.blur_strength,
        blur_expansion=args.blur_expansion,
        min_ellipse_size=args.min_ellipse_size,
        max_detections=args.max_detections,
        input_size=args.input_size,
        device=args.device,
        deepsort_max_age=args.max_age,
        deepsort_min_hits=args.min_hits,
        deepsort_iou_tracker=args.iou_tracker,
        enable_cuda_decoder=not args.disable_cuda_decoder,
        enable_cuda_encoder=not args.disable_cuda_encoder,
        output_codec=args.output_codec,
        output_bitrate=args.output_bitrate,
        skip_frames=max(1, args.skip_frames),
        enable_async_processing=args.async_processing,
        classes=args.classes,
        no_blur_track_ids=args.no_blur_track_ids,
        track_id_font_scale=args.track_id_font_scale,
        track_id_font_thickness=args.track_id_font_thickness,
        track_id_bg_alpha=args.track_id_bg_alpha
    )
    app = YOLOv9DeepSortApp(config)
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    if args.output:
        print("  'r' - Start recording (if --output specified and not already recording)")
        print("  's' - Stop recording")
    else:
        print("  Note: Recording ('r'/'s') requires --output to be specified")
    print(f"Processing every {config.skip_frames} frame(s)")
    print(f"Async processing: {'enabled' if config.enable_async_processing else 'disabled'}")
    print(f"Display: {'enabled' if not args.no_display else 'disabled'}")
    if config.no_blur_track_ids:
        print(f"No blur track IDs: {config.no_blur_track_ids}")
    print(f"Track ID font scale: {config.track_id_font_scale}")
    print(f"Track ID font thickness: {config.track_id_font_thickness}")
    print(f"Track ID background alpha: {config.track_id_bg_alpha}")
    print("\nStarting application...")
    try:
        app.run_yolov9_deepsort(args.video, args.output, display=not args.no_display)
    except Exception as e:
        print(f"Execution error: {e}")
        logging.error(f"Application error: {e}")
        sys.exit(1)
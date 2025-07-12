from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    video: str
    output: Optional[str] = None
    face_model: str = "yolov9e-face.onnx"
    person_model: str = "yolov9c.onnx"
    plate_model: str = "yolo-v9-t-640-license-plate-end2end.onnx"
    providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    skip_frames: int = 1

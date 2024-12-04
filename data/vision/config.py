from dataclasses import dataclass, field
from typing import Dict


@dataclass
class YOLOConfig:
    config_path: str = "vision/models/cross-hands.cfg"
    weights_path: str = "vision/models/cross-hands.weights"
    size: int = 416
    confidence: float = 0.5
    threshold: float = 0.3
    scale_factor: float = 1.2

    NETWORK_CONFIGS: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "normal": {
                "config": "vision/models/cross-hands.cfg",
                "weights": "vision/models/cross-hands.weights",
            },
            "tiny": {
                "config": "vision/models/cross-hands-tiny.weights",
                "weights": "vision/models/cross-hands-tiny.cfg",
            },
            "prn": {
                "config": "vision/models/cross-hands-tiny-prn.cfg",
                "weights": "vision/models/cross-hands-tiny-prn.weights",
            },
            "v4-tiny": {
                "config": "vision/models/cross-hands-yolov4-tiny.cfg",
                "weights": "vision/models/cross-hands-yolov4-tiny.weights",
            },
        }
    )


@dataclass
class SignLanguageConfig:
    model_path: str = "vision/models/sign_language.pth"
    num_classes: int = 29
    input_size: int = 128


@dataclass
class DetectorConfig:
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    sign_language: SignLanguageConfig = field(default_factory=SignLanguageConfig)
    predictions_dir: str = "predictions"
    char_buffer_duration: float = 3.0  # Duration in seconds to buffer characters
    NETWORK_CONFIGS: Dict = field(default_factory=dict)

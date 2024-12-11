from dataclasses import dataclass, field
from typing import Dict


@dataclass
class YOLOConfig:
    config_path: str = "src/vision/models/cross-hands.cfg"
    weights_path: str = "src/vision/models/cross-hands.weights"
    size: int = 416
    confidence: float = 0.5
    threshold: float = 0.3
    scale_factor: float = 1.2

    NETWORK_CONFIGS: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "normal": {
                "config": "src/vision/models/cross-hands.cfg",
                "weights": "src/vision/models/cross-hands.weights",
            },
            "tiny": {
                "config": "src/vision/models/cross-hands-tiny.weights",
                "weights": "src/vision/models/cross-hands-tiny.cfg",
            },
            "prn": {
                "config": "src/vision/models/cross-hands-tiny-prn.cfg",
                "weights": "src/vision/models/cross-hands-tiny-prn.weights",
            },
            "v4-tiny": {
                "config": "src/vision/models/cross-hands-yolov4-tiny.cfg",
                "weights": "src/vision/models/cross-hands-yolov4-tiny.weights",
            },
        }
    )


@dataclass
class SignLanguageConfig:
    model_path: str = "src/vision/models/sign_language_model_best.pth"
    num_classes: int = 29
    input_size: int = 128


@dataclass
class DetectorConfig:
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    sign_language: SignLanguageConfig = field(default_factory=SignLanguageConfig)
    predictions_dir: str = "predictions"
    char_buffer_duration: float = 3.0  # Duration in seconds to buffer characters
    min_detection_count: int = 10
    NETWORK_CONFIGS: Dict = field(default_factory=dict)

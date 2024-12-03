import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Tuple
import os

from .config import DetectorConfig
from .yolo import YOLOHandDetector
from .sign_language import SignLanguageClassifier

logger = logging.getLogger(__name__)


class SignLanguageDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self._setup_detectors()
        self._setup_output_dir()

    def _setup_detectors(self):
        # Initialize detectors with config
        self.hand_detector = YOLOHandDetector(self.config.yolo)
        self.sign_classifier = SignLanguageClassifier(self.config.sign_language)

    def _setup_output_dir(self):
        Path(self.config.predictions_dir).mkdir(exist_ok=True)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        try:
            # Detect hands
            hand_detections = self.hand_detector.detect_hands(frame)

            predicted_letter = None
            for x, y, w, h, conf in hand_detections:
                # Extract hand region
                hand_image = frame[y : y + h, x : x + w]
                if hand_image.size == 0:
                    continue

                # Classify hand sign
                predicted_letter = self.sign_classifier.predict(hand_image)

                # Draw detection
                self._draw_detection(frame, x, y, w, h, predicted_letter, conf)

            return frame, predicted_letter

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return frame, None

    @staticmethod
    def _draw_detection(
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        letter: Optional[str],
        confidence: float,
    ):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if letter:
            label = f"{letter} ({confidence:.2f})"
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

    def run_demo(self):
        """Run real-time demo using webcam feed."""
        cap = cv2.VideoCapture(0)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, prediction = self.process_frame(frame)
                cv2.imshow("Sign Language Detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        except Exception as e:
            logger.error(f"Demo execution failed: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Sign Language Detection Demo")
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="normal",
        choices=["normal", "tiny", "prn", "v4-tiny"],
        help="Specify the YOLO model to use for hand detection",
    )
    args = parser.parse_args()

    # Load configuration
    config = DetectorConfig()

    # Update configuration with CLI arguments
    model_config = config.yolo.NETWORK_CONFIGS[args.yolo_model]
    config.yolo.config_path = model_config["config"]
    config.yolo.weights_path = model_config["weights"]

    # Initialize detector with updated config
    detector = SignLanguageDetector(config)

    # Run demo
    detector.run_demo()


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    main()

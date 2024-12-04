import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Tuple
import os
from collections import deque, Counter
import time

from .config import DetectorConfig
from .yolo import YOLOHandDetector
from .sign_language import SignLanguageClassifier

logger = logging.getLogger(__name__)


class SignLanguageDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.char_buffer = deque()
        self.current_sentence = []
        self._setup_detectors()
        self._setup_output_dir()

    def _setup_detectors(self):
        self.hand_detector = YOLOHandDetector(self.config.yolo)
        self.sign_classifier = SignLanguageClassifier(self.config.sign_language)

    def _setup_output_dir(self):
        Path(self.config.predictions_dir).mkdir(exist_ok=True)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Process a single frame and return annotated frame with prediction."""
        try:
            hand_detections = self.hand_detector.detect_hands(frame)
            predicted_letter = None

            for x, y, w, h, conf in hand_detections:
                hand_image = frame[y : y + h, x : x + w]
                if hand_image.size == 0:
                    continue

                predicted_letter = self.sign_classifier.predict(hand_image)
                self._draw_detection(frame, x, y, w, h, predicted_letter, conf)

            # Draw the current sentence on the frame
            self._draw_sentence(frame)

            return frame, predicted_letter

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return frame, None

    def _draw_sentence(self, frame: np.ndarray) -> None:
        """Draw the current sentence on the frame."""
        sentence = "".join(self.current_sentence)
        cv2.putText(
            frame,
            sentence,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def process_character(self, predicted_char: Optional[str]) -> Optional[str]:
        """Process a predicted character and return completed sentence if available."""
        if predicted_char is None:
            return None

        # Update character buffer with (timestamp, char) tuple
        current_time = time.time()
        self.char_buffer.append((current_time, predicted_char))

        # Remove old detections
        self._clean_buffer(current_time)

        # Process buffer and return sentence if complete
        return self._process_buffer()

    def get_current_sentence(self) -> str:
        """Get the current sentence."""
        return "".join(self.current_sentence)

    def _clean_buffer(self, current_time: float) -> None:
        """Remove detections older than char_buffer_duration."""
        while (
            self.char_buffer
            and current_time - self.char_buffer[0][0] > self.config.char_buffer_duration
        ):
            self.char_buffer.popleft()

    def _process_buffer(self) -> Optional[str]:
        """Process the character buffer and return completed sentence if available."""
        if not self.char_buffer:
            return None

        # Count only the characters, ignoring timestamps
        char_counts = Counter(char for _, char in self.char_buffer)
        most_common_char, count = char_counts.most_common(1)[0]

        if count < self.config.min_detection_count:
            return None

        self.char_buffer.clear()

        if most_common_char in ["nothing", "unknown"]:
            return None

        if most_common_char == "del":  # Termination character
            sentence = "".join(self.current_sentence)
            self.current_sentence = []  # Reset for next sentence
            return sentence
        elif most_common_char == "space":
            self.current_sentence.append(" ")
        else:
            self.current_sentence.append(most_common_char)

        return None

    @staticmethod
    def _draw_detection(
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        letter: Optional[str],
        confidence: float,
    ) -> None:
        """Draw bounding box and label on the frame."""
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if letter:
            label = f"{letter} ({confidence:.2f})"
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

    def run_demo(self) -> None:
        """Run the real-time sign language detection demo."""
        cap = cv2.VideoCapture(0)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame and get prediction
                annotated_frame, prediction = self.process_frame(frame)
                cv2.imshow("Sign Language Detection", annotated_frame)

                # Process character and check for completed sentence
                sentence = self.process_character(prediction)
                if sentence:
                    logger.info(f"Sentence: {sentence}")

                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

        except Exception as e:
            logger.error(f"Demo execution failed: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Sign Language Detection Demo")
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="normal",
        choices=["normal", "tiny", "prn", "v4-tiny"],
        help="YOLO model type for hand detection",
    )
    args = parser.parse_args()

    # Initialize configuration
    config = DetectorConfig()
    model_config = config.yolo.NETWORK_CONFIGS[args.yolo_model]
    config.yolo.config_path = model_config["config"]
    config.yolo.weights_path = model_config["weights"]

    # Run demo
    detector = SignLanguageDetector(config)
    detector.run_demo()


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    main()

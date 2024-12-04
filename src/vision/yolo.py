import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path

from .config import YOLOConfig

logger = logging.getLogger(__name__)


class YOLOHandDetector:
    def __init__(self, config: YOLOConfig):
        self.config = config
        self._setup_network()

    def _setup_network(self):
        try:
            config_path = Path(self.config.config_path)
            weights_path = Path(self.config.weights_path)

            if not config_path.exists() or not weights_path.exists():
                raise FileNotFoundError("YOLO config or weights file not found")

            self.net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))

            layer_names = self.net.getLayerNames()
            self.output_names = [
                layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
            ]

        except Exception as e:
            logger.error(f"Failed to initialize YOLO network: {e}")
            raise

    def detect_hands(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect hands in image and return their bounding boxes and confidence scores.
        Returns: List of (x, y, w, h, confidence) tuples
        """
        try:
            ih, iw = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                image,
                1 / 255.0,
                (self.config.size, self.config.size),
                swapRB=True,
                crop=False,
            )

            self.net.setInput(blob)
            detections = self.net.forward(self.output_names)

            # Process detections
            boxes = []
            confidences = []
            for output in detections:
                for detection in output:
                    confidence = detection[5]
                    if confidence > self.config.confidence:
                        center_x, center_y, width, height = (
                            detection[0:4] * np.array([iw, ih, iw, ih])
                        ).astype(int)

                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
                        scaled_bbox = self.scale_bbox((x, y, int(width), int(height)))
                        boxes.append(scaled_bbox)
                        confidences.append(float(confidence))

            # Apply Non-Maximum Suppression
            idxs = cv2.dnn.NMSBoxes(
                boxes, confidences, self.config.confidence, self.config.threshold
            )

            filtered_boxes = []
            if len(idxs) > 0:
                for i in idxs.flatten():
                    # Extract the bounding box coordinates
                    x, y, w, h = boxes[i]
                    confidence = confidences[i]

                    # Append the result as needed
                    filtered_boxes.append((x, y, w, h, confidence))

            return filtered_boxes

        except Exception as e:
            logger.error(f"Hand detection failed: {e}")
            return []

    def scale_bbox(self, bbox):
        """Scale the bounding box by a factor while keeping the center fixed"""
        x, y, w, h = bbox

        # Calculate center
        center_x = x + w / 2
        center_y = y + h / 2

        # Scale width and height using config value
        new_w = w * self.config.scale_factor
        new_h = h * self.config.scale_factor

        # Calculate new top-left corner
        new_x = center_x - new_w / 2
        new_y = center_y - new_h / 2

        # Ensure coordinates stay within image bounds
        new_x = int(max(0, new_x))
        new_y = int(max(0, new_y))
        new_w = int(new_w)
        new_h = int(new_h)

        return [new_x, new_y, new_w, new_h]

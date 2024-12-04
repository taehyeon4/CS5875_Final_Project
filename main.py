<<<<<<< HEAD
from src import nlpmodule, cvmodule

# computer vision part
# cvinstance = cvmodule()

#temp final string
final_string = "my name is akhil and i fucking hate this pizza it tastes so ass"

#language processing part
nlp_instance = nlpmodule.NLPModule()
processed_string = nlp_instance.process_text(final_string)
print(f"Processed text: {processed_string}")
=======
import argparse
import logging
import os

import cv2

from vision.config import DetectorConfig
from vision.demo import SignLanguageDetector

logger = logging.getLogger(__name__)


class Demo:
    def __init__(self, detector_config: DetectorConfig):
        self.detector_config = detector_config
        self.detector = SignLanguageDetector(self.detector_config)

    def run(self) -> None:
        """Run the real-time sign language detection demo."""
        cap = cv2.VideoCapture(0)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame and get prediction
                annotated_frame, prediction = self.detector.process_frame(frame)
                cv2.imshow("Sign Language Detection", annotated_frame)

                # Process character and check for completed sentence
                sentence = self.detector.process_character(prediction)
                if sentence:
                    logger.info(f"Sentence: {sentence}")

                # TODO: Add logic to handle the sentence

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

    demo = Demo(config)
    demo.run()


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    main()
>>>>>>> 864417985a06baf5bfc436b749af2586573eeeb1

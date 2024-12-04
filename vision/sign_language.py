import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from .config import SignLanguageConfig

logger = logging.getLogger(__name__)


class SignLanguageModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=5, stride=2, padding=2, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class SignLanguageClassifier:
    def __init__(self, config: SignLanguageConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model()
        self._setup_transform()

    def _setup_model(self):
        self.model = SignLanguageModel(self.config.num_classes).to(self.device)
        self._load_model()
        self.model.eval()

    def _setup_transform(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.input_size, self.config.input_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _load_model(self):
        try:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")

            state_dict = torch.load(model_path, map_location=self.device)

            # Add "model." prefix to keys if missing
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith("model."):
                    new_state_dict["model." + k] = v
                else:
                    new_state_dict[k] = v

            # Load the modified state dictionary
            self.model.load_state_dict(new_state_dict)
            logger.info(f"Model loaded successfully from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @torch.no_grad()
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Optional[str]:
        try:
            # Handle different input types
            if isinstance(image, str):
                image = Image.open(image).convert("L")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("L")
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")

            processed_image = self.transform(image).unsqueeze(0).to(self.device)
            output = self.model(processed_image)
            _, predicted = torch.max(output, 1)

            return self.get_label(predicted.item())

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    @staticmethod
    def get_label(index: int) -> str:
        labels = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "H",
            8: "I",
            9: "J",
            10: "K",
            11: "L",
            12: "M",
            13: "N",
            14: "O",
            15: "P",
            16: "Q",
            17: "R",
            18: "S",
            19: "T",
            20: "U",
            21: "V",
            22: "W",
            23: "X",
            24: "Y",
            25: "Z",
            26: "del",
            27: "nothing",
            28: "space",
        }
        return labels.get(index, "unknown")

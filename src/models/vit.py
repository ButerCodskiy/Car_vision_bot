"""Vision Transformer модель для классификации автомобилей."""

from pathlib import Path
from typing import Sequence

import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor


CAR_CLASSES = ["Audi", "Bentley", "BMW", "Porsche", "Toyota"]
DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"


class VitModel:
    def __init__(self, model_path: str, device: torch.device, class_names: Sequence[str]) -> None:
        self.device = device
        self.class_names: list[str] = list(class_names)
        self.model = self._load_model(model_path)
        self.processor = self._load_processor()

    def class_name(self, idx: int) -> str:
        return self.class_names[idx]

    def _load_model(self, model_path: str) -> ViTForImageClassification:
        model = ViTForImageClassification.from_pretrained(
            DEFAULT_MODEL_NAME,
            num_labels=len(self.class_names),
            ignore_mismatched_sizes=True,
        )

        weights = torch.load(Path(model_path), map_location=self.device)
        if isinstance(weights, dict):
            weights = weights.get("model_state_dict") or weights.get("state_dict") or weights

        model.load_state_dict(weights, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _load_processor(self) -> ViTImageProcessor:
        try:
            return ViTImageProcessor.from_pretrained(DEFAULT_MODEL_NAME)
        except Exception:
            return ViTFeatureExtractor.from_pretrained(DEFAULT_MODEL_NAME)


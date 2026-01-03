"""Сервис предсказаний для моделей классификации автомобилей."""
import asyncio
import time
from typing import Tuple

import torch
from PIL import Image

from src.config import settings
from src.models.vit import CAR_CLASSES, VitModel


class PredictionService:
    """Отвечает за инференс ViT модели."""

    def __init__(self, vit_model: VitModel) -> None:
        self.vit = vit_model

    def predict(self, image: Image.Image) -> Tuple[str, float, float]:
        start = time.time()
        encoding = self.vit.processor(image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(self.vit.device)

        with torch.no_grad():
            logits = self.vit.model(pixel_values=pixel_values).logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            class_idx = probs.argmax().item()
            confidence = probs[class_idx].item() * 100

        elapsed = (time.time() - start) * 1000
        return self.vit.class_name(class_idx), confidence, elapsed


prediction_service = PredictionService(
    VitModel(
        model_path=settings.VIT_MODEL_PATH,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        class_names=CAR_CLASSES,
    )
)


async def predict_vit_async(image: Image.Image) -> Tuple[str, float, float]:
    """
    Асинхронный wrapper для predict_vit.
    Выполняет предсказание в отдельном потоке, не блокируя event loop.

    Args:
        image: PIL Image в формате RGB

    Returns:
        Tuple[str, float, float]: (название класса, вероятность в процентах, время в мс)
    """
    return await asyncio.to_thread(prediction_service.predict, image)
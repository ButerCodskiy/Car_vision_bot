"""Функции для работы с изображениями из bytes и конвертации в PIL Image."""

from io import BytesIO

from PIL import Image


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Загружает изображение из bytes и конвертирует в PIL Image.

    Args:
        image_bytes: Байты изображения (например, из Telegram)

    Returns:
        PIL Image в формате RGB

    Raises:
        ValueError: Если не удалось загрузить изображение
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        # Конвертируем в RGB, если изображение в другом формате (например, RGBA, L)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Не удалось загрузить изображение из bytes: {e}") from e


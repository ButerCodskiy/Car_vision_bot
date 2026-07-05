import aiohttp
from src.config import settings

async def get_prediction_from_api(image_bytes: bytes) -> dict:
    """
    Отправляет картинку в FastAPI сервис и возвращает ответ в формате словаря.
    """
    url = f"{settings.API_URL}/predict"
    
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        # Добавляем байты файла в форму, чтобы FastAPI UploadFile корректно спарсил
        data.add_field('file', image_bytes, filename='image.jpg', content_type='image/jpeg')
        
        async with session.post(url, data=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise ValueError(f"API Error {response.status}: {error_text}")

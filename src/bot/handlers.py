from aiogram import Router, types
from aiogram.filters import CommandStart
from aiogram import F

from src.services.api_client import get_prediction_from_api
import logging


router = Router()


@router.message(CommandStart())
async def start_cmd(
    message: types.Message,
):
    return await message.answer(
        "Привет! 👋\n\n"
        "Я могу распознать одну из 5 марок автомобилей:\n"
        "<b>Audi, Bentley, BMW, Porsche, Toyota.</b>\n\n"
        "Пришлите <b>фотографию</b> автомобиля, и я дам прогноз от двух нейросетей"
    )


@router.message(F.photo)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    
    file = await message.bot.download(photo)
    image_bytes = file.read()
    
    try:
        # Отправляем байты картинки в наш новый FastAPI микросервис
        result = await get_prediction_from_api(image_bytes)
        
        await message.answer(
            "📊 Результаты распознавания:\n\n"
            f"🤖 Vision Transformer:\n"
            f"   Марка: {result['brand']}\n"
            f"   Вероятность: {result['confidence']:.0f}%\n"
            f"   Время обработки (API): {result['inference_time_ms']:.0f} ms\n\n"
            "📩 Присылайте новые фото, и я дам прогноз."
        )
    except Exception as e:
        logging.error(f"API Prediction failed: {e}")
        await message.answer("⚠️ К сожалению, сервис распознавания сейчас недоступен. Попробуйте позже.")


@router.message(F.text)
async def handle_text(message: types.Message):
    await message.answer(
        "Пришлите <b>фотографию</b> автомобиля\n\n"
    )

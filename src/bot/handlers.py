from aiogram import Router, types
from aiogram.filters import CommandStart
from aiogram import F

from src.services.predict import predict_vit_async
from src.utils.image_loader import load_image_from_bytes


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
    image = load_image_from_bytes(image_bytes)
    
    brand, confidence, inference_time = await predict_vit_async(image)
    
    await message.answer(
        "📊 Результаты распознавания:\n\n"
        f"🤖 Vision Transformer:\n"
        f"   Марка: {brand}\n"
        f"   Вероятность: {confidence:.0f}%\n"
        f"   Время обработки: {inference_time:.0f} ms\n\n"
        "📩 Присылайте новые фото, и я дам прогноз."
    )


@router.message(F.text)
async def handle_text(message: types.Message):
    await message.answer(
        "Пришлите <b>фотографию</b> автомобиля\n\n"
    )

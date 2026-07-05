from fastapi import FastAPI, UploadFile, File, HTTPException
import io
from PIL import Image

from src.services.predict import predict_vit_async

app = FastAPI(
    title="Car Classification ML API",
    description="API для распознавания марок автомобилей (ViT)",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    """Эндпоинт для проверки жизнеспособности сервиса."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_car(file: UploadFile = File(...)):
    """
    Эндпоинт предсказания. Принимает картинку (jpeg, png, etc.)
    и возвращает марку машины, уверенность и время инференса.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Читаем байты из загруженного файла
        contents = await file.read()
        # Загружаем PIL Image (используем ту же логику, что была в боте)
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Запускаем асинхронный инференс
        brand, confidence, inference_time = await predict_vit_async(image)
        
        return {
            "success": True,
            "brand": brand,
            "confidence": confidence,
            "inference_time_ms": inference_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

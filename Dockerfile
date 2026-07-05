FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей, полезных для работы с изображениями
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Прописываем PYTHONPATH, чтобы Python видел пакет src
ENV PYTHONPATH=/app

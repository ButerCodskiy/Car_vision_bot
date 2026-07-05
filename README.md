# 🚗 Car Classification System (Computer Vision)

**Микросервисное ML-приложение для классификации марок автомобилей**, использующее современные методы Computer Vision.

Проект представляет собой законченное ML-решение: от обучения моделей (Vision Transformer и ResNet50) с использованием Transfer Learning до деплоя в продакшен через микросервисную архитектуру (FastAPI + Streamlit + Telegram Bot) в Docker.

Поддерживаемые классы: `Audi`, `Bentley`, `BMW`, `Porsche`, `Toyota`.

---

## 🏗 Архитектура

Проект разделен на 3 независимых микросервиса:
1. **ML Backend (FastAPI)** — REST API сервис, загружающий в память веса нейросети и выполняющий асинхронный инференс (предсказание) по поступающим изображениям.
2. **Telegram Bot (Aiogram 3)** — асинхронный бот-клиент, пересылающий фотографии пользователей в ML Backend.
3. **Web Frontend (Streamlit)** — интерактивный веб-интерфейс для загрузки фотографий через браузер.

---

## 📊 Сравнение моделей и Результаты

В ходе работы над проектом были обучены и сравнены две архитектуры. Основной целью было выявить баланс между точностью распознавания и скоростью обучения.

**Используемые методы:**
*   **Transfer Learning:** Использование предобученных весов (ImageNet).
*   **Fine-tuning:** Двухэтапное обучение (сначала обучение классификатора с замороженным "хребтом", затем разморозка и дообучение всей сети с низким Learning Rate).
*   **Augmentation:** RandomRotation, HorizontalFlip, ColorJitter.

### Итоговые метрики на тестовой выборке

| Metric | ResNet50 (Baseline) | Vision Transformer (ViT) |
| :--- | :---: | :---: |
| **Test Accuracy** | 85.79% | **92.89%** 🏆 |
| **Test F1-Score** (Weighted) | 0.8551 | **0.9287** 🏆 |
| **Training Time** | 46.26 min | **35.1 min** ⚡ |
| **Model Size** | **~90 MB** 💾 | 327.4 MB |

> **Вывод:** Модель **Vision Transformer (ViT-base-patch16-224)** показала значительно более высокую точность (~93%) и F1-score по сравнению с ResNet50 (~86%). Несмотря на больший размер весов, ViT обучался быстрее и показал лучшую обобщающую способность, поэтому именно он был выбран для продакшена.

---

## 🛠 Технический стек

*   **ML & Data Science**: PyTorch, HuggingFace Transformers, Torchvision, Scikit-learn, Pandas.
*   **Backend API**: FastAPI, Uvicorn, aiohttp.
*   **Frontend**: Streamlit.
*   **Telegram Client**: Aiogram 3.x.
*   **Инфраструктура**: Docker, Docker Compose, Git LFS.

---

## 🚀 Установка и запуск (Через Docker)

Проект использует Git LFS для хранения весов модели. Убедитесь, что он установлен перед клонированием.

### 1. Клонирование репозитория
```bash
git lfs install
git clone https://github.com/ВАШ_НИК/car-classification-bot.git
cd car-classification-bot
```

### 2. Конфигурация
Создайте файл `.env` в корне проекта и укажите токен вашего бота:
```env
BOT_TOKEN=your_telegram_bot_token_here
```

### 3. Запуск всех сервисов (Docker Compose)
```bash
docker-compose up -d --build
```

После запуска сервисы будут доступны по следующим адресам:
- **FastAPI (Swagger)**: `http://localhost:8000/docs`
- **Streamlit Web UI**: `http://localhost:8501`
- **Telegram Bot**: Запустится в фоне автоматически.

---

## 📂 Структура проекта

```text
car-classification-bot/
├─ notebooks/                            # Jupyter-ноутбуки (Обучение ViT / ResNet)
├─ src/
│  ├─ api/
│  │  └─ main.py                         # Точка входа FastAPI (ML Backend)
│  ├─ bot/
│  │  ├─ __init__.py                     # Инициализация бота
│  │  └─ handlers.py                     # Пересылка фото от юзера в API
│  ├─ web/
│  │  └─ app.py                          # Streamlit Frontend
│  ├─ models/
│  │  ├─ vit.py                          # Класс-обертка для ViT
│  │  └─ best_vit_model_stage1.pth       # Веса модели (Git LFS)
│  ├─ services/
│  │  ├─ predict.py                      # Асинхронный сервис инференса
│  │  └─ api_client.py                   # Клиент aiohttp для обращения к FastAPI
│  ├─ main.py                            # Точка входа Telegram-бота
│  └─ config.py                          # Конфигурация (Pydantic)
├─ docker-compose.yml                    # Оркестрация сервисов
├─ Dockerfile                            # Docker-образ приложения
├─ requirements.txt                      # Зависимости
└─ README.md
```

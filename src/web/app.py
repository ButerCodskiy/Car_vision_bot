# pyrefly: ignore [missing-import]
import streamlit as st
import requests
from PIL import Image
import os

# Получаем URL API. По умолчанию используем локальный адрес FastAPI сервера
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Car Classification", page_icon="🚗")

st.title("🚗 Распознавание марок автомобилей")
st.write("Загрузите фотографию автомобиля, и нейросеть (Vision Transformer) определит её марку.")
st.write("Поддерживаемые классы: **Audi, Bentley, BMW, Porsche, Toyota**.")

uploaded_file = st.file_uploader("Выберите фото...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Отображаем загруженную картинку
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_container_width=True)
    
    if st.button("Распознать марку"):
        with st.spinner('Модель анализирует изображение...'):
            try:
                # Подготавливаем файл для отправки
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Отправляем POST запрос в наш FastAPI микросервис
                response = requests.post(f"{API_URL}/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Успешно распознано!")
                    
                    st.write("### 📊 Результаты")
                    st.write(f"**🚗 Марка:** {result['brand']}")
                    st.write(f"**🎯 Уверенность:** {result['confidence']:.2f}%")
                    st.write(f"**⏱ Время обработки API:** {result['inference_time_ms']:.0f} ms")
                else:
                    st.error(f"Ошибка сервера API: {response.status_code}")
                    st.write(response.text)
            except requests.exceptions.ConnectionError:
                st.error(f"Не удалось подключиться к API по адресу {API_URL}. Убедитесь, что FastAPI сервер запущен.")

FROM python:3.11-slim

# --- Системные зависимости ---
# Добавляем curl (для healthcheck), ffmpeg, libgl, libglib — всё, что нужно для YOLO и OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
 && rm -rf /var/lib/apt/lists/*

# --- Рабочая директория ---
WORKDIR /app

# --- Устанавливаем Python-зависимости ---
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Копируем код приложения ---
COPY backend/app ./app

# --- Логи и порты ---
EXPOSE 8000

# --- Запуск FastAPI + логирование ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]

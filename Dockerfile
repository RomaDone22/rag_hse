FROM python:3.12

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Открываем порт для API
EXPOSE 8585

# Запускаем приложение
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8585"]

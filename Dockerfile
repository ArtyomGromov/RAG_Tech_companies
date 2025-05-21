# Используем легковесный образ Python
FROM python:3.10-slim

# Предотвращаем буферизацию вывода
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt ./

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код бота
COPY app.py ./

# Открываем порт для webhook
EXPOSE 80

# Запускаем FastAPI через uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

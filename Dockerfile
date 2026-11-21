FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p artifacts && python src/train.py

EXPOSE 5000

CMD ["gunicorn", "api.server:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4"]

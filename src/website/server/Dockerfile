FROM python:3.12-slim

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ /app
WORKDIR /app

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
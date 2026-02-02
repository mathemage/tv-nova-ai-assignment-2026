FROM python:3.11-slim

WORKDIR /app

# Build from repo root: docker build -f service/Dockerfile -t novatv-model .
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY service/main.py /app/
# Optional: copy a default model so container runs without mount (placeholder)
# RUN mkdir -p /app/models

ENV MODEL_DIR=/app/models
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

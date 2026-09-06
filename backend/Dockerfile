# CrackCatch backend - FastAPI + the detection pipeline.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend:/app/model

# libGL and libglib are needed by OpenCV even in headless builds; curl backs
# the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so a source edit does not invalidate the slow torch layer.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY model/crackcatch_model /app/model/crackcatch_model
COPY model/scripts /app/model/scripts

# Trained weights are mounted at runtime (see docker-compose.yml) so the image
# stays small and a retrained checkpoint needs no rebuild.
RUN mkdir -p /app/model/weights /app/storage/snapshots /app/storage/uploads \
             /app/storage/repairs /app/storage/retraining

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]

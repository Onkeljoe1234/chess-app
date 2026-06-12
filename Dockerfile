FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY predictors/ predictors/
COPY models/ models/
COPY templates/ templates/
COPY static/ static/

ENV ORT_THREADS=4
# Set APP_PASSWORD at runtime; empty disables auth.
ENV APP_PASSWORD=""

EXPOSE 8000
# 1 worker, threads for concurrent games: the predictor (model session, eval
# cache, per-game trees) is shared in-process state and must not be forked.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]

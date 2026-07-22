# All-nano build (2026-07-22): the app talks ONLY to the self-contained
# nano UCI binaries — no torch, no ONNX runtime, no chesscore wheel. One
# build stage compiles the two portable engine binaries (250K-dims and
# 1M-dims) against the image's glibc (the host's glibc version is moot);
# x86-64-v3 baseline — the Strato host has AVX2/FMA/BMI2, no VNNI.

# --- stage 1: the nano binaries ---------------------------------------------
FROM python:3.12-slim AS nano-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && rm -rf /var/lib/apt/lists/*
COPY engine-repo/nano /src/nano
RUN cd /src/nano && make app app1m

# --- stage 2: the app --------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the engine repo subset (engine/ python package + nano weights; the app
# imports engine.app_models / engine.nano_predictor — both torch-free)
COPY engine-repo/ /opt/cct/
COPY --from=nano-build /src/nano/nano_app /opt/cct/nano/nano_app
COPY --from=nano-build /src/nano/nano_app_1m /opt/cct/nano/nano_app_1m
ENV CCT_ENGINE_PATH=/opt/cct
ENV NANO_THREADS=4

COPY app.py .
COPY templates/ templates/
COPY static/ static/

EXPOSE 8000
# 1 worker, threads for concurrent games: the predictors (per-game engine
# processes) are shared in-process state and must not be forked.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]

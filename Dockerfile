# All-nano build (2026-07-22): the app talks ONLY to the self-contained
# nano UCI binaries — no torch, no ONNX runtime, no chesscore wheel. One
# build stage compiles the three portable engine binaries (250K-dims,
# 1M-dims, 90K-edge dims) against the image's glibc (the host's glibc version is moot);
# x86-64-v3 baseline — the Strato host has AVX2/FMA/BMI2, no VNNI.

# --- stage 1: the nano binaries ---------------------------------------------
FROM python:3.12-slim AS nano-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && rm -rf /var/lib/apt/lists/*
COPY engine-repo/nano /src/nano
RUN cd /src/nano && make app app1m app90k

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
COPY --from=nano-build /src/nano/nano_app_90k /opt/cct/nano/nano_app_90k
ENV CCT_ENGINE_PATH=/opt/cct
# 6 = the host's thread count (EPYC-Milan VM, 3c/6t): measured 2026-08-19
# 6,285 vs 4,484 evals/s (T6 vs T4) on the deployed Pico; concurrent
# games share the budget inside the predictor
ENV NANO_THREADS=6
# finished games are archived here as PGN — mount a volume (deploy.sh does)
# or the archive dies with the container
ENV GAMES_DIR=/data/games
VOLUME ["/data/games"]

COPY app.py .
COPY templates/ templates/
COPY static/ static/

EXPOSE 8000
# 1 worker, threads for concurrent games: the predictors (per-game engine
# processes) are shared in-process state and must not be forked.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]

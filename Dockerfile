# Two-stage build: chesscore (Rust MCTS core) is compiled against the image's
# python, then the app runs on the real engine repo (synced by deploy.sh into
# ./engine-repo, mounted at /opt/cct — see CCT_ENGINE_PATH in app.py).

# --- stage 1: the chesscore wheel -------------------------------------------
FROM python:3.12-slim AS chesscore-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential && rm -rf /var/lib/apt/lists/*
# rustup respects engine-repo/chesscore/rust-toolchain.toml (pinned 1.96)
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin
COPY engine-repo/chesscore /src/chesscore
RUN cd /src/chesscore && maturin build --release -i python3.12 -o /wheels
# nano: portable build (x86-64-v3 — the Strato host has AVX2/FMA/BMI2, no
# VNNI), compiled against the IMAGE's glibc so the host version is moot
COPY engine-repo/nano /src/nano
RUN cd /src/nano && make app

# --- stage 2: the app --------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=chesscore-build /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# the engine repo subset (engine/, config.py, data_processing/, onnx_models/)
COPY engine-repo/ /opt/cct/
COPY --from=chesscore-build /src/nano/nano_app /opt/cct/nano/nano_app
ENV CCT_ENGINE_PATH=/opt/cct
ENV NANO_THREADS=4

COPY app.py .
COPY templates/ templates/
COPY static/ static/

ENV ORT_THREADS=4

EXPOSE 8000
# 1 worker, threads for concurrent games: the predictor (model session, eval
# cache, per-game trees) is shared in-process state and must not be forked.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]

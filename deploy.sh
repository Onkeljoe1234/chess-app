#!/bin/bash
set -e

SERVER="root@85.214.195.22"
SSH_KEY="/home/loupmut2/.ssh/id_rsa"
REMOTE_DIR="/var/www/vhosts/datascience-bonn.de/chess-app"
CONTAINER_NAME="chess-app"
IMAGE_NAME="chess-app"
PORT=8000   # bound to 127.0.0.1 only - reachable solely through the Plesk proxy

echo "=== Deploying chess.datascience-bonn.de ==="

# 0. Stage the engine-repo subset the Docker build needs (see Dockerfile).
#    MODEL_FILE: override via env to ship a different checkpoint.
ENGINE_SRC="${ENGINE_SRC:-/home/loupmut2/Dev/compact_chess_transformers}"
MODEL_FILE="${MODEL_FILE:-chess_transformer_m=CANON65_FEAT16_legalmask_ds=M.nncf-int8.onnx}"
STAGE=/home/loupmut2/Dev/chess-app/engine-repo
echo "Staging engine subset (model: $MODEL_FILE)..."
rm -rf "$STAGE"
mkdir -p "$STAGE/onnx_models"
rsync -rq --exclude '__pycache__' \
  "$ENGINE_SRC/engine" "$ENGINE_SRC/data_processing" "$STAGE/"
rsync -rq --exclude target --exclude '__pycache__' "$ENGINE_SRC/chesscore" "$STAGE/"
cp "$ENGINE_SRC/config.py" "$STAGE/"
cp "$ENGINE_SRC/onnx_models/$MODEL_FILE" "$STAGE/onnx_models/"
# nano: sources (built in-container against the image glibc — the host
# runs glibc 2.31, dev-box binaries would not load) + the 250k weights
mkdir -p "$STAGE/nano"
cp "$ENGINE_SRC"/nano/{nano.c,chess.c,search.c,chess.h,nano_math.h,Makefile,cct250k.bin} "$STAGE/nano/"

# 1. Sync files to server
echo "Syncing files..."
rsync -zruh \
  --exclude '.git' \
  --exclude '.vscode' \
  --exclude '.env' \
  --exclude 'venv' \
  --exclude '__pycache__' \
  -e "ssh -i $SSH_KEY" \
  /home/loupmut2/Dev/chess-app/ \
  "$SERVER:$REMOTE_DIR/"

# 2. Build and restart container on server
echo "Building and restarting container..."
ssh -i "$SSH_KEY" "$SERVER" <<ENDSSH
cd $REMOTE_DIR

if [ ! -f .env ]; then
  echo "ERROR: $REMOTE_DIR/.env missing on the server."
  echo "Create it once with:  echo 'APP_PASSWORD=<your-password>' > $REMOTE_DIR/.env"
  exit 1
fi

# Build FIRST — a failed build must not take the running app down.
docker build -t $IMAGE_NAME .

docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# 127.0.0.1 binding: the app is NOT exposed publicly; Plesk's nginx proxies
# https://chess.datascience-bonn.de -> localhost:$PORT
# MODEL_PATH: the staged MODEL_FILE is also the runtime model (CANON* models
# are auto-wrapped by the engine; see engine/app_predictor.py).
docker run -d \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -p 127.0.0.1:$PORT:8000 \
  --env-file .env \
  -e MODEL_PATH=/opt/cct/onnx_models/$MODEL_FILE \
  $IMAGE_NAME

sleep 2
docker logs $CONTAINER_NAME --tail 5
ENDSSH

echo "=== Deploy complete! ==="
echo "App: http://127.0.0.1:$PORT on the server -> proxy via chess.datascience-bonn.de"

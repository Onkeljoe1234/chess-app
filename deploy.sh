#!/bin/bash
set -e

SERVER="root@85.214.195.22"
SSH_KEY="/home/loupmut2/.ssh/id_rsa"
REMOTE_DIR="/var/www/vhosts/datascience-bonn.de/chess-app"
CONTAINER_NAME="chess-app"
IMAGE_NAME="chess-app"
PORT=8000   # bound to 127.0.0.1 only - reachable solely through the Plesk proxy

echo "=== Deploying chess.datascience-bonn.de ==="

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

docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

docker build -t $IMAGE_NAME .

# 127.0.0.1 binding: the app is NOT exposed publicly; Plesk's nginx proxies
# https://chess.datascience-bonn.de -> localhost:$PORT
docker run -d \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -p 127.0.0.1:$PORT:8000 \
  --env-file .env \
  $IMAGE_NAME

sleep 2
docker logs $CONTAINER_NAME --tail 5
ENDSSH

echo "=== Deploy complete! ==="
echo "App: http://127.0.0.1:$PORT on the server -> proxy via chess.datascience-bonn.de"

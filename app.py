"""Chess app: play against the compact transformer (ONNX, CPU) with MCTS.

- One opponent: the <1M-param transformer with batched MCTS; the strength
  slider sets NN evaluations per move (0 = greedy policy).
- Multi-game: each /start returns a game_id; concurrent games don't clobber
  each other (the old version had one global board).
- Auth: HTTP Basic with a shared password from APP_PASSWORD (empty = no auth).
"""
import hmac
import os
import random
import secrets
import time
from pathlib import Path

import chess
from flask import Flask, Response, jsonify, render_template, request

from predictors import MCTSPredictor

BASE_DIR = Path(__file__).parent / "models"
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "chess_transformer_m=XXXS_V26_ds=M.QUInt8.onnx"))
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
ORT_THREADS = int(os.getenv("ORT_THREADS", "4"))
MAX_NODES = 6000
MAX_GAMES = 32

predictor = MCTSPredictor(
    model_path=MODEL_PATH,
    fen_vocab_path=BASE_DIR / "fen_vocab.json",
    move_vocab_path=BASE_DIR / "move_vocab.json",
    threads=ORT_THREADS,
)

app = Flask(__name__)
games = {}  # game_id -> {"board": chess.Board, "player_color": bool, "ts": float}


@app.before_request
def check_auth():
    if not APP_PASSWORD:
        return None
    auth = request.authorization
    if auth and auth.password and hmac.compare_digest(auth.password, APP_PASSWORD):
        return None
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="chess"'})


def _evict_old_games():
    while len(games) > MAX_GAMES:
        oldest = min(games, key=lambda g: games[g]["ts"])
        games.pop(oldest, None)
        predictor.drop_game(oldest)


def _nodes_from_request(data) -> int:
    try:
        nodes = int(data.get("nodes", 1000))
    except (TypeError, ValueError):
        nodes = 1000
    return max(0, min(MAX_NODES, nodes))


def _game_status(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"Checkmate! {winner} wins."
    if board.is_stalemate():
        return "Stalemate! The game is a draw."
    if board.is_insufficient_material():
        return "Draw: Insufficient material."
    if board.can_claim_draw():
        return "Draw (repetition / 50-move rule)."
    return "Game over."


def _ai_info_json(info: dict) -> dict:
    return {
        "win_prob_white": round(info.get("win_prob_white", 0.5), 4),
        "nodes": info.get("nodes", 0),
        "depth": info.get("depth", 0),
        "time": round(info.get("time", 0.0), 2),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_game():
    data = request.json or {}
    color = data.get("color", "white")
    nodes = _nodes_from_request(data)

    player_color = chess.WHITE if color == "white" else \
        chess.BLACK if color == "black" else random.choice([chess.WHITE, chess.BLACK])
    game_id = secrets.token_hex(8)
    board = chess.Board()
    games[game_id] = {"board": board, "player_color": player_color, "ts": time.time()}
    _evict_old_games()

    ai = None
    if board.turn != player_color:
        move, info = predictor.get_move(board, nodes=nodes, game_id=game_id)
        if move:
            board.push(move)
            ai = {"uci": move.uci(), **_ai_info_json(info)}

    return jsonify({
        "game_id": game_id,
        "fen": board.fen(),
        "player_color": "white" if player_color == chess.WHITE else "black",
        "ai": ai,
    })


@app.route("/move", methods=["POST"])
def make_move():
    """Apply the player's move only - returns immediately so the UI shows the
    completed move (castling rook, en-passant capture) BEFORE the engine
    starts thinking. The client then calls /ai for the reply."""
    data = request.json or {}
    game = games.get(data.get("game_id"))
    if game is None:
        return jsonify({"error": "Unknown game - start a new one"}), 404
    board = game["board"]
    game["ts"] = time.time()

    if board.turn != game["player_color"]:
        return jsonify({"error": "Not your turn"}), 400
    try:
        move = chess.Move.from_uci(data.get("move", ""))
    except (ValueError, chess.InvalidMoveError):
        return jsonify({"error": "Bad move format"}), 400
    if move not in board.legal_moves:
        return jsonify({"error": "Illegal move"}), 400

    board.push(move)
    over = board.is_game_over(claim_draw=True)
    return jsonify({
        "fen": board.fen(),
        "game_over": over,
        "status": _game_status(board) if over else None,
    })


@app.route("/ai", methods=["POST"])
def ai_move():
    data = request.json or {}
    game = games.get(data.get("game_id"))
    if game is None:
        return jsonify({"error": "Unknown game - start a new one"}), 404
    board = game["board"]
    game["ts"] = time.time()

    if board.turn == game["player_color"] or board.is_game_over(claim_draw=True):
        return jsonify({"error": "Not the engine's turn"}), 400

    move, info = predictor.get_move(board, nodes=_nodes_from_request(data),
                                    game_id=data.get("game_id"))
    ai = None
    if move:
        board.push(move)
        ai = {"uci": move.uci(), **_ai_info_json(info)}

    over = board.is_game_over(claim_draw=True)
    return jsonify({
        "fen": board.fen(),
        "game_over": over,
        "status": _game_status(board) if over else None,
        "ai": ai,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)

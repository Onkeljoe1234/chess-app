"""Chess app: play against the compact transformer (ONNX, CPU) with MCTS.

- One opponent: the <1M-param transformer with batched MCTS; the strength
  slider sets NN evaluations per move (0 = greedy policy).
- Multi-game: each /start returns a game_id; concurrent games don't clobber
  each other (the old version had one global board).
- Auth: HTTP Basic with a shared password from APP_PASSWORD (empty = no auth).

Engine boundary: the app talks to the engine repo (compact_chess_transformers,
mounted at CCT_ENGINE_PATH) through exactly one class — EnginePredictor with
get_move(board, nodes, game_id) / drop_game(game_id). Search behavior, model
format and quantization live entirely on the engine side.
"""
import hmac
import os
import random
import secrets
import sys
import time

import chess
from flask import Flask, Response, jsonify, render_template, request

CCT_ENGINE_PATH = os.getenv("CCT_ENGINE_PATH", "/opt/cct")
sys.path.insert(0, CCT_ENGINE_PATH)
from engine.app_predictor import EnginePredictor  # noqa: E402

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(
    CCT_ENGINE_PATH, "onnx_models",
    "chess_transformer_m=CANON65_FEAT16_legalmask_ds=M.nncf-int8.onnx"))
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
ORT_THREADS = int(os.getenv("ORT_THREADS", "4"))
MAX_NODES = {"nano": 15000, "big": 6000}   # nano runs ~4x the evals/s
MAX_GAMES = 32

# Two engines, lazily constructed. "nano" (default): the 250k model inside
# the self-contained nano binary (~0.5 MB engine+weights), opening variety
# on. "big": the 1M CANON65 ship model through the Python/ONNX stack.
NANO_BIN = os.getenv("NANO_BIN", os.path.join(CCT_ENGINE_PATH, "nano", "nano_app"))
NANO_WEIGHTS = os.getenv("NANO_WEIGHTS",
                         os.path.join(CCT_ENGINE_PATH, "nano", "cct250k.bin"))
NANO_THREADS = int(os.getenv("NANO_THREADS", "4"))
NANO_VARIETY = float(os.getenv("NANO_VARIETY", "0.7"))
DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", "nano")
ENGINES = ("nano", "big")

_predictors = {}


def get_predictor(name: str):
    if name not in _predictors:
        if name == "nano":
            from engine.nano_predictor import NanoPredictor
            _predictors["nano"] = NanoPredictor(
                NANO_BIN, NANO_WEIGHTS, threads=NANO_THREADS, variety=NANO_VARIETY)
        else:
            _predictors["big"] = EnginePredictor(MODEL_PATH, threads=ORT_THREADS)
    return _predictors[name]


def _engine_from_request(data) -> str:
    e = data.get("engine", DEFAULT_ENGINE)
    return e if e in ENGINES else DEFAULT_ENGINE

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
        for p in _predictors.values():
            p.drop_game(oldest)


def _nodes_from_request(data, engine: str) -> int:
    try:
        nodes = int(data.get("nodes", 1000))
    except (TypeError, ValueError):
        nodes = 1000
    return max(0, min(MAX_NODES.get(engine, 6000), nodes))


def _game_end(board: chess.Board) -> dict:
    """Structured end-of-game info for the client banner."""
    if board.is_checkmate():
        reason, winner = "checkmate", ("white" if board.turn == chess.BLACK else "black")
    elif board.is_stalemate():
        reason, winner = "stalemate", None
    elif board.is_insufficient_material():
        reason, winner = "insufficient", None
    elif board.can_claim_draw():
        reason, winner = "draw_rule", None
    else:
        reason, winner = "over", None
    return {"reason": reason, "winner": winner, "plies": len(board.move_stack)}


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
    engine = _engine_from_request(data)
    nodes = _nodes_from_request(data, engine)
    player_color = chess.WHITE if color == "white" else \
        chess.BLACK if color == "black" else random.choice([chess.WHITE, chess.BLACK])
    game_id = secrets.token_hex(8)
    board = chess.Board()
    games[game_id] = {"board": board, "player_color": player_color,
                      "engine": engine, "ts": time.time()}
    _evict_old_games()

    ai = None
    if board.turn != player_color:
        move, info = get_predictor(engine).get_move(board, nodes=nodes, game_id=game_id)
        if move:
            board.push(move)
            ai = {"uci": move.uci(), **_ai_info_json(info)}

    return jsonify({
        "game_id": game_id,
        "fen": board.fen(),
        "player_color": "white" if player_color == chess.WHITE else "black",
        "engine": engine,
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
        "end": _game_end(board) if over else None,
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

    eng = game.get("engine", DEFAULT_ENGINE)
    move, info = get_predictor(eng).get_move(
        board, nodes=_nodes_from_request(data, eng), game_id=data.get("game_id"))
    ai = None
    if move:
        board.push(move)
        ai = {"uci": move.uci(), **_ai_info_json(info)}

    over = board.is_game_over(claim_draw=True)
    return jsonify({
        "fen": board.fen(),
        "game_over": over,
        "end": _game_end(board) if over else None,
        "ai": ai,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)

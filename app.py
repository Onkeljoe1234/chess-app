"""Chess app: play against the compact-transformer family (nano engine, CPU).

- Four tiers (Nano 961,942 / Pico 236,182 / Femto 137,046 / Atto 87,478
  exact params), all on the self-contained nano UCI engine — the app is
  ONNX-free. The strength slider sets NN evaluations per move; greedy is
  not offered (per-tier min-node floors, enforced engine-side too).
- Multi-game: each /start returns a game_id; concurrent games don't clobber
  each other.
- Auth: HTTP Basic with a shared password from APP_PASSWORD (empty = no auth).

Engine boundary: the app talks to the engine repo (compact_chess_transformers,
mounted at CCT_ENGINE_PATH) through engine.app_models.make_predictor —
NanoPredictor with get_move(board, nodes, game_id) / drop_game(game_id).
Search behavior, model format and quantization live on the engine side.
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
from engine.app_models import TIERS, make_predictor  # noqa: E402

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
MAX_GAMES = 32
NANO_THREADS = int(os.getenv("NANO_THREADS", "4"))
NANO_VARIETY = float(os.getenv("NANO_VARIETY", "0.7"))
DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", "pico")

# All four tiers run on the nano engine (the app is ONNX-free since the 1M
# model's nano port). Engine-side truth (exact params, min_nodes, binaries)
# lives in engine.app_models; ONLY app/UI concerns live here: per-tier node
# caps & defaults (caps ~match wall-clock — the 1M model runs ~1/4 the
# evals/s). The UI stays minimal: the select shows name + exact params.
TIER_UI = {
    "nano":  dict(max_nodes=6000, def_nodes=4000),
    "pico":  dict(max_nodes=15000, def_nodes=10000),
    "femto": dict(max_nodes=15000, def_nodes=10000),
    "atto":  dict(max_nodes=15000, def_nodes=10000),
}
ENGINES = tuple(TIERS)
_LEGACY = {"big": "nano"}          # pre-catalog client keys

_predictors = {}


def get_predictor(name: str):
    if name not in _predictors:
        _predictors[name] = make_predictor(
            name, threads=NANO_THREADS, variety=NANO_VARIETY)
    return _predictors[name]


def tier_payload():
    """Static tier metadata for the template (select + model info box)."""
    return [{"key": k, "display": t.display, "params": t.params,
             "min_nodes": t.min_nodes, "node_step": t.node_step,
             **TIER_UI[k]} for k, t in TIERS.items()]


def _engine_from_request(data) -> str:
    e = data.get("engine", DEFAULT_ENGINE)
    e = _LEGACY.get(e, e)
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
    # tier floor (greedy is not offered — the predictors floor too) and cap
    return max(TIERS[engine].min_nodes,
               min(TIER_UI[engine]["max_nodes"], nodes))


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
    return render_template("index.html", tiers=tier_payload(),
                           default_engine=DEFAULT_ENGINE)


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

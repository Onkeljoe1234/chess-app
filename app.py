"""Chess app: play against the compact-transformer family (nano engine, CPU).

- One tier since 2026-10-16: Femto (94,198 params, the 90K RayScan edge
  point; Pico/Nano retired — Femto plays Pico to parity at 37 % of the
  params and ~2.2x the speed), on the self-contained nano UCI engine — the app is
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
import datetime
import hmac
import io
import os
import random
import secrets
import sys
import time

import chess
import chess.pgn
from flask import Flask, Response, jsonify, render_template, request, send_file

CCT_ENGINE_PATH = os.getenv("CCT_ENGINE_PATH", "/opt/cct")
sys.path.insert(0, CCT_ENGINE_PATH)
from engine.app_models import TIERS, make_predictor  # noqa: E402

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
MAX_GAMES = 32
# finished games are archived as PGN, one file per game_id (a replayed
# game overwrites its own file), exportable as one bundle via /games.pgn.
# Container: mount a volume at /data/games or the archive dies with the
# container (see deploy.sh).
GAMES_DIR = os.getenv("GAMES_DIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "games"))
NANO_THREADS = int(os.getenv("NANO_THREADS", "4"))
NANO_VARIETY = float(os.getenv("NANO_VARIETY", "0.7"))
DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", "femto")

# All tiers run on the nano engine (the app is ONNX-free since the 1M
# model's nano port). Engine-side truth (exact params, min_nodes, binaries)
# lives in engine.app_models; ONLY app/UI concerns live here: per-tier node
# caps & defaults (caps ~match wall-clock — the 1M model runs ~1/4 the
# evals/s). The UI stays minimal: the select shows name + exact params.
TIER_UI = {
    # Femto = the 90K edge point: ~29k+ evals/s on the server (6T, AVX2 int8)
    "femto": dict(max_nodes=60000, def_nodes=20000),
}
ENGINES = tuple(TIERS)
# retired / pre-catalog client keys -> Femto (lineup 2026-10-16)
_LEGACY = {"big": "femto", "nano": "femto", "pico": "femto"}

_predictors = {}


def get_predictor(name: str):
    if name not in _predictors:
        _predictors[name] = make_predictor(
            name, threads=NANO_THREADS, variety=NANO_VARIETY)
    return _predictors[name]


def _kb(path: str) -> int:
    try:
        return round(os.path.getsize(path) / 1024)
    except OSError:
        return 0


def tier_payload():
    """Static tier metadata for the template. Sizes are measured from the
    ACTUAL files (in-container: the staged weights and the binaries built
    against the image glibc) — no hand-maintained byte counts."""
    return [{"key": k, "display": t.display, "params": t.params,
             "min_nodes": t.min_nodes, "node_step": t.node_step,
             "weights_kb": _kb(t.weights_path), "engine_kb": _kb(t.binary_path),
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


def _can_undo(game: dict) -> bool:
    """True while there is a PLAYER ply to take back. Playing black, the
    engine's opening move alone is not undoable."""
    board = game["board"]
    if not board.move_stack:
        return False
    return len(board.move_stack) >= (1 if game["player_color"] == chess.WHITE else 2)


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
        "depth": info.get("depth", 0),          # PV length (most-visited path)
        "seldepth": info.get("seldepth", 0),    # deepest descent of the search
        "time": round(info.get("time", 0.0), 2),
    }


# --- game archive ---------------------------------------------------------

def _result_string(game: dict, end: dict) -> str:
    if end.get("resigned"):
        return "0-1" if game["player_color"] == chess.WHITE else "1-0"
    w = end.get("winner")
    return "1-0" if w == "white" else "0-1" if w == "black" else "1/2-1/2"


def archive_game(game_id: str, game: dict, end: dict):
    """Write the finished game as PGN (one file per game_id — replaying the
    same game overwrites its own file, so an undo-and-refinish leaves ONE
    record, not two)."""
    board = game["board"]
    if not board.move_stack:
        return
    pgn = chess.pgn.Game.from_board(board)
    tier = TIERS[game.get("engine", DEFAULT_ENGINE)]
    player_white = game["player_color"] == chess.WHITE
    started = datetime.datetime.fromtimestamp(game.get("started", time.time()))
    pgn.headers["Event"] = "Compact Chess Transformer"
    pgn.headers["Site"] = "chess.datascience-bonn.de"
    pgn.headers["Date"] = started.strftime("%Y.%m.%d")
    pgn.headers["Round"] = "-"
    engine_name = f"CCT {tier.display} ({tier.params} params)"
    pgn.headers["White"] = "Mensch" if player_white else engine_name
    pgn.headers["Black"] = engine_name if player_white else "Mensch"
    pgn.headers["Result"] = _result_string(game, end)
    pgn.headers["Termination"] = "resignation" if end.get("resigned") \
        else end.get("reason", "over")
    pgn.headers["Annotator"] = f"nodes={game.get('nodes', 0)}"
    try:
        os.makedirs(GAMES_DIR, exist_ok=True)
        path = os.path.join(GAMES_DIR, f"{int(started.timestamp())}_{game_id}.pgn")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(pgn) + "\n")
        game["archived"] = path
    except OSError as exc:                    # never break play over archiving
        app.logger.warning("could not archive game %s: %s", game_id, exc)


def _finish(game_id: str, game: dict, end: dict) -> dict:
    """Archive a finished game. Idempotent: the path is derived from the
    game_id, so an undo + re-finish overwrites its own record."""
    if end:
        archive_game(game_id, game, end)
    return end


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
                      "engine": engine, "ts": time.time(),
                      "started": time.time(), "nodes": nodes}
    _evict_old_games()

    # The engine's first move is NOT computed here: the client must be able
    # to flip/redraw the board first (playing black used to look like a
    # freeze while /start blocked on the search). engine_to_move tells the
    # client to render, then call /ai — the same two-phase flow as /move.
    return jsonify({
        "game_id": game_id,
        "fen": board.fen(),
        "player_color": "white" if player_color == chess.WHITE else "black",
        "engine": engine,
        "engine_to_move": board.turn != player_color,
        "ai": None,
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
        "end": _finish(data["game_id"], game, _game_end(board)) if over else None,
        "can_undo": _can_undo(game),
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
    game["nodes"] = _nodes_from_request(data, eng)
    move, info = get_predictor(eng).get_move(
        board, nodes=game["nodes"], game_id=data.get("game_id"))
    ai = None
    if move:
        board.push(move)
        ai = {"uci": move.uci(), **_ai_info_json(info)}

    over = board.is_game_over(claim_draw=True)
    return jsonify({
        "fen": board.fen(),
        "game_over": over,
        "end": _finish(data["game_id"], game, _game_end(board)) if over else None,
        "ai": ai,
        "can_undo": _can_undo(game),
    })


@app.route("/undo", methods=["POST"])
def undo_move():
    """Take back one full move (the player's ply plus the engine's reply if
    it already answered), repeatable down to the game's start position. The
    engine's own opening move as White is kept — undoing it would leave a
    position the player cannot act on.

    nano's tree reuse is prefix-based: a SHORTER move list fails the prefix
    check engine-side and starts a fresh tree, so no cache surgery is needed
    here (verified in search.c).
    """
    data = request.json or {}
    game_id = data.get("game_id")
    game = games.get(game_id)
    if game is None:
        return jsonify({"error": "Unknown game - start a new one"}), 404
    board = game["board"]
    game["ts"] = time.time()

    if not _can_undo(game):
        return jsonify({"error": "Nichts zum Zurücknehmen"}), 400
    board.pop()                                   # the last ply on the board
    if board.move_stack and board.turn != game["player_color"]:
        board.pop()                               # ...and the player's own
    game.pop("resigned", None)                    # a taken-back game runs on
    return jsonify({
        "fen": board.fen(),
        "game_over": False,
        "can_undo": _can_undo(game),
    })


@app.route("/resign", methods=["POST"])
def resign():
    data = request.json or {}
    game_id = data.get("game_id")
    game = games.get(game_id)
    if game is None:
        return jsonify({"error": "Unknown game - start a new one"}), 404
    board = game["board"]
    game["ts"] = time.time()
    game["resigned"] = True
    winner = "black" if game["player_color"] == chess.WHITE else "white"
    end = {"reason": "resign", "winner": winner, "resigned": True,
           "plies": len(board.move_stack)}
    return jsonify({"fen": board.fen(), "game_over": True,
                    "end": _finish(game_id, game, end)})


@app.route("/games.pgn")
def games_bundle():
    """Every archived game as ONE PGN file — importable in any analysis GUI
    (SCID, ChessBase, Lichess study import)."""
    parts, count = [], 0
    if os.path.isdir(GAMES_DIR):
        for name in sorted(os.listdir(GAMES_DIR)):
            if not name.endswith(".pgn"):
                continue
            try:
                with open(os.path.join(GAMES_DIR, name), encoding="utf-8") as f:
                    parts.append(f.read().strip())
                count += 1
            except OSError:
                continue
    if not count:
        return Response("Noch keine beendeten Partien gespeichert.\n",
                        mimetype="text/plain")
    body = ("\n\n".join(parts) + "\n").encode("utf-8")
    stamp = datetime.datetime.now().strftime("%Y%m%d")
    return send_file(io.BytesIO(body), mimetype="application/x-chess-pgn",
                     as_attachment=True,
                     download_name=f"cct-partien-{stamp}-{count}.pgn")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

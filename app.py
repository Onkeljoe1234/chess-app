from flask import Flask, render_template, request, jsonify
import chess
import os
import random
from pathlib import Path

# --- Predictor Imports ---
from predictors import RandomPredictor, StockfishPredictor, TransformerMultitaskPredictor


STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/games/stockfish")
BASE_DIR = Path("models")

predictors = {
    "random": RandomPredictor(),
    "stockfish": StockfishPredictor(stockfish_path=STOCKFISH_PATH),
    "transformer_mt": TransformerMultitaskPredictor(
        model_path=BASE_DIR / "chess_transformer_multi_m=XXS_ds=XXL.pt",
        fen_vocab_path=BASE_DIR / "fen_vocab.json",
        move_vocab_path=BASE_DIR / "move_vocab.json"
    )
}

app = Flask(__name__)
board = chess.Board()

# Simple in-memory storage for game state. For a real app, use sessions.
game_state = {
    "player_color": chess.WHITE,
    "predictor": predictors["transformer_mt"] # Default predictor
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start_game():
    board.reset()
    
    # 1. Get settings from the frontend
    engine_name = request.json.get("engine")
    player_color_str = request.json.get("color")

    # 2. Set the predictor engine
    game_state["predictor"] = predictors.get(engine_name, predictors["random"])
    print(game_state["predictor"], flush=True)
    
    # 3. Set player color
    if player_color_str == "white":
        game_state["player_color"] = chess.WHITE
    elif player_color_str == "black":
        game_state["player_color"] = chess.BLACK
    else: # Random
        game_state["player_color"] = random.choice([chess.WHITE, chess.BLACK])
    
    # 4. If AI is White, make the first move
    ai_move_uci = None
    if board.turn != game_state["player_color"]:
        ai_move = game_state["predictor"].get_move(board)
        if ai_move:
            board.push(ai_move)
            ai_move_uci = ai_move.uci()

    return jsonify({
        "fen": board.fen(),
        "player_color": "white" if game_state["player_color"] == chess.WHITE else "black",
        "ai_first_move": ai_move_uci
    })


    
@app.route("/move", methods=["POST"])
def make_move():
    try:
        # Prevent moves if it's not the player's turn
        if board.turn != game_state["player_color"]:
             return jsonify({"error": "Not your turn"}), 400

        move_uci = request.json.get("move")
        move = chess.Move.from_uci(move_uci)
  
        if move in board.legal_moves:
            board.push(move)

            if board.is_game_over():
                return jsonify({"fen": board.fen(), "game_over": True, "status": get_game_status()})

            # --- Add this line for backend logging ---
            print(f"AI is thinking with: {game_state['predictor']}", flush=True)
            
            # Get AI's response move
            ai_move = game_state["predictor"].get_move(board)
            if ai_move:
                board.push(ai_move)

            if board.is_game_over():
                return jsonify({"fen": board.fen(), "game_over": True, "status": get_game_status()})

            return jsonify({"fen": board.fen(), "game_over": False})
        else:
            return jsonify({"error": "Illegal move"}), 400
    except Exception as e:
        print(f"Error in /move: {e}")
        return jsonify({"error": str(e)}), 500

  
    
@app.route("/switch", methods=["POST"])
def switch_model():
    engine_name = request.json.get("engine")
    
    if engine_name in predictors:
        game_state["predictor"] = predictors[engine_name]
        # Log the switch to the backend console
        print(f"--- Switched predictor to: {game_state['predictor']} ---", flush=True)
        return jsonify({"success": True, "new_engine": engine_name})
    else:
        return jsonify({"error": "Invalid engine name"}), 400

  

def get_game_status():
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"Checkmate! {winner} wins."
    if board.is_stalemate(): return "Stalemate! The game is a draw."
    if board.is_insufficient_material(): return "Draw: Insufficient material."
    # Add other draw conditions if you like
    return "Game over."

if __name__ == "__main__":
    app.run(debug=True)
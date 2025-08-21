import chess
from .base_predictor import Predictor
from stockfish import Stockfish

class StockfishPredictor(Predictor):
    """A predictor that uses the Stockfish engine to find the best move."""

    def __init__(self, stockfish_path: str = "/usr/local/bin/stockfish"):
        """
        Initializes the Stockfish engine.
        Make sure to update the path to your Stockfish executable.
        """
        try:
            self.stockfish = Stockfish(path=stockfish_path)
        except FileNotFoundError:
            print(f"Error: Stockfish executable not found at {stockfish_path}")
            # You might want to fall back to another predictor or handle this more gracefully
            raise

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Uses Stockfish to determine the best move.
        """
        self.stockfish.set_fen_position(board.fen())
        best_move = self.stockfish.get_best_move()
        return chess.Move.from_uci(best_move)
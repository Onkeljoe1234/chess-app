    
import random
import chess
from .base_predictor import Predictor

class RandomPredictor(Predictor):
    """A predictor that returns a random legal move."""

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Selects a random legal move from the current board state.
        """
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

  
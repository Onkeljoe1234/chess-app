    
from abc import ABC, abstractmethod
import chess

class Predictor(ABC):
    """Abstract base class for a chess move predictor."""

    @abstractmethod
    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Given a board state, return the best move.

        :param board: The current chess.Board object.
        :return: A chess.Move object representing the predicted move.
        """
        pass

  
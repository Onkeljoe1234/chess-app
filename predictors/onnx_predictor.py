# onnx_predictor.py
import json
from typing import Optional

import chess
import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is required. Install with:\n"
        "  pip install onnxruntime          # CPU\n"
        "  pip install onnxruntime-gpu      # NVIDIA GPU"
    ) from e

from .base_predictor import Predictor


# --- Helpers ---

def fen_to_tokens_fixed77(fen: str) -> list[str]:
    board, stm, castling, ep, halfmove, fullmove = fen.split()
    expanded = []
    for ch in board:
        if ch == '/':
            continue
        if ch.isdigit():
            expanded.extend(['.'] * int(ch))
        else:
            expanded.append(ch)
    assert len(expanded) == 64

    toks = expanded + [stm]

    def has(c): return '1' if c in castling else '0'
    toks += [has('K'), has('Q'), has('k'), has('q')]

    if ep != '-':
        toks += [ep[0], ep[1]]
    else:
        toks += ['-', '-']

    toks += list(f"{int(halfmove):03d}")
    toks += list(f"{int(fullmove):03d}")
    assert len(toks) == 77
    return toks


def _pick_greedy_legal(logits: np.ndarray, legal_ids: np.ndarray) -> int:
    legal_logits = logits[legal_ids]
    return int(legal_ids[int(np.argmax(legal_logits))])

def _log_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable log_softmax for a 1D numpy array."""
    c = np.max(x)
    return x - c - np.log(np.sum(np.exp(x - c)))


# --- ONNX-only predictor ---

class ONNXMultitaskPredictor(Predictor):
    """
    Predictor that runs an exported ONNX model (policy+value) using onnxruntime.
    Includes a powerful, optional NegaMax search for stronger play.
    """
    def __init__(
        self,
        model_path: str,
        fen_vocab_path: str,
        move_vocab_path: str,
        providers: Optional[list[str]] = None,
        # --- Search Configuration ---
        use_negamax_search: bool = True,
        beam_size: int = 8,
        branch_factor: int = 3,
        # --- ONNX Session Fine-tuning ---
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ):
        self.name = "ONNX Multitask"
        
        # --- Search Parameters ---
        self.use_negamax_search = use_negamax_search
        self.beam_size = beam_size
        self.branch_factor = branch_factor
        
        if self.use_negamax_search:
             print(f"NegaMax search enabled (candidates={self.beam_size}, reply_branches={self.branch_factor})")
        else:
            print("Using simple greedy prediction.")

        # --- Load vocabs ---
        with open(fen_vocab_path, 'r') as f:
            self.fen_vocab = json.load(f)
        with open(move_vocab_path, 'r') as f:
            self.move_vocab = json.load(f)
        self.move_vocab_inv = {v: k for k, v in self.move_vocab.items()}
        self.max_seq_len = 77

        # --- Choose providers ---
        available = ort.get_available_providers()
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        self.device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        print(f"ONNX providers: {providers}")

        # --- Session options ---
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads is not None: so.intra_op_num_threads = int(intra_op_num_threads)
        if inter_op_num_threads is not None: so.inter_op_num_threads = int(inter_op_num_threads)

        # --- Create session ---
        self.ort_session = ort.InferenceSession(model_path, sess_options=so, providers=providers)

        # --- Cache IO names and indices ---
        self._onnx_input_name = self._infer_input_name(self.ort_session)
        self._policy_out_index = self._guess_policy_output_index(self.ort_session)
        self._value_out_index = self._guess_value_output_index(self.ort_session)

        print(f"ONNX loaded. Input='{self._onnx_input_name}', "
              f"PolicyIndex={self._policy_out_index}, ValueIndex={self._value_out_index}")

    # --------- IO inference helpers ---------

    @staticmethod
    def _infer_input_name(session) -> str:
        # Prefer a friendly 'input_ids' name if present; else first input.
        for inp in session.get_inputs():
            if inp.name == "input_ids": return "input_ids"
        return session.get_inputs()[0].name

    @staticmethod
    def _guess_policy_output_index(session) -> int:
        """Heuristically identify the policy output (likely large vocab size)."""
        outs = session.get_outputs()
        # Prefer names with 'policy' or 'logit'
        for i, o in enumerate(outs):
            if any(k in o.name.lower() for k in ["policy", "logit"]): return i
        # Fallback: pick the 2D output with the largest last dimension.
        best_i, best_dim = -1, -1
        for i, o in enumerate(outs):
            if o.shape and len(o.shape) == 2 and o.shape[1] > best_dim:
                best_dim = o.shape[1]
                best_i = i
        return best_i if best_i != -1 else 0

    @staticmethod
    def _guess_value_output_index(session) -> int:
        """Heuristically identify the value output (likely shape [B, 1])."""
        outs = session.get_outputs()
        # Prefer names with 'value'
        for i, o in enumerate(outs):
            if "value" in o.name.lower(): return i
        # Fallback: pick the 2D output with the smallest last dimension (usually 1)
        best_i, best_dim = -1, float('inf')
        for i, o in enumerate(outs):
            if o.shape and len(o.shape) == 2 and o.shape[1] < best_dim:
                best_dim = o.shape[1]
                best_i = i
        # If policy and value are somehow identical, pick the one not chosen for policy
        policy_idx = ONNXMultitaskPredictor._guess_policy_output_index(session)
        return best_i if best_i != policy_idx else 1 - policy_idx

    # --------- Inference Wrappers ---------

    def _tokenize_ids_np(self, board: chess.Board) -> np.ndarray:
        tokens = fen_to_tokens_fixed77(board.fen())
        unk_id = self.fen_vocab.get('<UNK>', 0)
        return np.array([[self.fen_vocab.get(t, unk_id) for t in tokens]], dtype=np.int64)

    def _predict_policy_and_value(self, input_ids_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_session.run(None, {self._onnx_input_name: input_ids_np})
        return outputs[self._policy_out_index], outputs[self._value_out_index]

    # --------- Public API and Search Implementation ---------

    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        if not any(board.legal_moves):
            return None

        if self.use_negamax_search:
            best_move_uci = self._negamax_search(board)
        else: # Fallback to simple greedy
            input_ids_np = self._tokenize_ids_np(board)
            policy_logits, _ = self._predict_policy_and_value(input_ids_np)
            logits = policy_logits[0]

            legal_uci = [m.uci() for m in board.legal_moves]
            legal_ids = np.fromiter((self.move_vocab[u] for u in legal_uci if u in self.move_vocab), dtype=np.int32)

            if legal_ids.size == 0:
                return np.random.choice(list(board.legal_moves))
            
            move_id = _pick_greedy_legal(logits, legal_ids)
            best_move_uci = self.move_vocab_inv[move_id]

        return chess.Move.from_uci(best_move_uci)

    def _negamax_search(self, board: chess.Board) -> str:
        """Predicts a move using a 2-ply NegaMax search with NumPy."""
        perspective = 1 if board.turn == chess.WHITE else -1

        # 1. Generate our best initial candidate moves
        input_ids_np = self._tokenize_ids_np(board)
        policy_logits, _ = self._predict_policy_and_value(input_ids_np)
        policy_logits = policy_logits.squeeze(0)

        legal_uci = [m.uci() for m in board.legal_moves if m.uci() in self.move_vocab]
        if not legal_uci:
            return list(board.legal_moves)[0].uci()

        legal_indices = np.array([self.move_vocab[u] for u in legal_uci], dtype=np.int32)
        legal_log_probs = _log_softmax(policy_logits[legal_indices])
        
        num_candidates = min(self.beam_size, len(legal_uci))
        # Use argpartition for efficiency, equivalent to topk
        topk_indices = np.argpartition(-legal_log_probs, num_candidates-1)[:num_candidates]
        candidate_moves = [legal_uci[i] for i in topk_indices]

        # 2. For each candidate, find the opponent's best replies
        boards_after_opp_reply = []
        move_indices_map = []

        boards_to_eval_opp = []
        moves_to_boards_map = [] 
        for i, uci in enumerate(candidate_moves):
            board_after_our_move = board.copy()
            board_after_our_move.push_uci(uci)
            start_idx = len(boards_to_eval_opp)
            if not board_after_our_move.is_game_over():
                boards_to_eval_opp.append(board_after_our_move)
            moves_to_boards_map.append((start_idx, len(boards_to_eval_opp)))

        if boards_to_eval_opp:
            batch_ids = np.vstack([self._tokenize_ids_np(b) for b in boards_to_eval_opp])
            opp_policy_logits, _ = self._predict_policy_and_value(batch_ids)

            for i, uci in enumerate(candidate_moves):
                start, end = moves_to_boards_map[i]
                if start == end: # Our move was game-ending
                    board_after_our_move = board.copy()
                    board_after_our_move.push_uci(uci)
                    boards_after_opp_reply.append(board_after_our_move)
                    move_indices_map.append(i)
                    continue

                logits = opp_policy_logits[start]
                board_after_our_move = boards_to_eval_opp[start]
                
                opp_legal_uci = [m.uci() for m in board_after_our_move.legal_moves if m.uci() in self.move_vocab]
                if not opp_legal_uci: continue

                opp_indices = np.array([self.move_vocab[u] for u in opp_legal_uci], dtype=np.int32)
                opp_log_probs = _log_softmax(logits[opp_indices])

                num_opp_replies = min(self.branch_factor, len(opp_legal_uci))
                top_opp_indices = np.argpartition(-opp_log_probs, num_opp_replies-1)[:num_opp_replies]

                for opp_idx in top_opp_indices:
                    final_board = board_after_our_move.copy()
                    final_board.push_uci(opp_legal_uci[opp_idx])
                    boards_after_opp_reply.append(final_board)
                    move_indices_map.append(i)

        if not boards_after_opp_reply:
            return candidate_moves[0] if candidate_moves else list(board.legal_moves)[0].uci()

        # 3. Batch evaluate all final positions
        final_batch_ids = np.vstack([self._tokenize_ids_np(b) for b in boards_after_opp_reply])
        _, final_values = self._predict_policy_and_value(final_batch_ids)

        # 4. Find the MIN value (opponent's best) for each of our moves
        initial_score = -2.0 * perspective
        move_scores = np.full(num_candidates, initial_score, dtype=np.float32)

        for i in range(len(boards_after_opp_reply)):
            original_move_index = move_indices_map[i]
            
            # *** THE FIX IS HERE ***
            # final_values[i] is already a scalar, so we remove the extra [0] index.
            value_from_our_perspective = float(final_values[i]) * perspective
            
            if move_scores[original_move_index] == initial_score or value_from_our_perspective < move_scores[original_move_index]:
                move_scores[original_move_index] = value_from_our_perspective

        # 5. Choose the move with the MAX score after the opponent's reply
        best_move_index = np.argmax(move_scores)
        return candidate_moves[best_move_index]

# # onnx_predictor.py
# import json
# from typing import Optional

# import chess
# import numpy as np

# try:
#     import onnxruntime as ort
# except ImportError as e:
#     raise ImportError(
#         "onnxruntime is required. Install with:\n"
#         "  pip install onnxruntime          # CPU\n"
#         "  pip install onnxruntime-gpu      # NVIDIA GPU"
#     ) from e

# from .base_predictor import Predictor


# # --- Helpers (same behavior as your original) ---

# def fen_to_tokens_fixed77(fen: str) -> list[str]:
#     board, stm, castling, ep, halfmove, fullmove = fen.split()
#     expanded = []
#     for ch in board:
#         if ch == '/':
#             continue
#         if ch.isdigit():
#             expanded.extend(['.'] * int(ch))
#         else:
#             expanded.append(ch)
#     assert len(expanded) == 64

#     toks = expanded + [stm]

#     def has(c): return '1' if c in castling else '0'
#     toks += [has('K'), has('Q'), has('k'), has('q')]

#     if ep != '-':
#         toks += [ep[0], ep[1]]
#     else:
#         toks += ['-', '-']

#     toks += list(f"{int(halfmove):03d}")
#     toks += list(f"{int(fullmove):03d}")
#     assert len(toks) == 77
#     return toks


# def _pick_greedy_legal(logits: np.ndarray, legal_ids: np.ndarray) -> int:
#     """Return the token id of the highest-logit legal move."""
#     legal_logits = logits[legal_ids]
#     return int(legal_ids[int(np.argmax(legal_logits))])


# # --- ONNX-only predictor ---

# class ONNXMultitaskPredictor(Predictor):
#     """
#     Predictor that runs an exported ONNX model (policy+value) using onnxruntime.

#     Assumes the ONNX model takes an input named 'input_ids' (int64, shape [B, 77]).
#     If exported with different names, the class will fall back to the first input and
#     heuristically pick the policy output.
#     """

#     def __init__(
#         self,
#         model_path: str,
#         fen_vocab_path: str,
#         move_vocab_path: str,
#         providers: Optional[list[str]] = None,
#         intra_op_num_threads: Optional[int] = None,
#         inter_op_num_threads: Optional[int] = None,
#     ):
#         self.name = "ONNX Multitask"
#         # --- Load vocabs ---
#         with open(fen_vocab_path, 'r') as f:
#             self.fen_vocab = json.load(f)
#         with open(move_vocab_path, 'r') as f:
#             self.move_vocab = json.load(f)
#         self.move_vocab_inv = {v: k for k, v in self.move_vocab.items()}
#         self.max_seq_len = 77

#         # --- Choose providers (default: CUDA if available, else CPU) ---
#         available = ort.get_available_providers()
#         if providers is None:
#             if "CUDAExecutionProvider" in available:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#                 self.device = "cuda"
#             else:
#                 providers = ["CPUExecutionProvider"]
#                 self.device = "cpu"
#         else:
#             self.device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
#         print(f"ONNX providers: {providers}")

#         # --- Session options ---
#         so = ort.SessionOptions()
#         so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#         if intra_op_num_threads is not None:
#             so.intra_op_num_threads = int(intra_op_num_threads)
#         if inter_op_num_threads is not None:
#             so.inter_op_num_threads = int(inter_op_num_threads)

#         # --- Create session ---
#         self.ort_session = ort.InferenceSession(model_path, sess_options=so, providers=providers)

#         # --- Cache IO names ---
#         self._onnx_input_name = self._infer_input_name(self.ort_session)
#         self._onnx_output_names = [o.name for o in self.ort_session.get_outputs()]
#         self._policy_out_index = self._guess_policy_output_index(self.ort_session, self._onnx_output_names)

#         print(f"ONNX loaded. Input='{self._onnx_input_name}', Outputs={self._onnx_output_names}, "
#               f"PolicyIndex={self._policy_out_index}")

#     # --------- IO inference helpers ---------

#     @staticmethod
#     def _infer_input_name(session) -> str:
#         # Prefer a friendly 'input_ids' name if present; else first input.
#         for inp in session.get_inputs():
#             if inp.name == "input_ids":
#                 return "input_ids"
#         return session.get_inputs()[0].name

#     @staticmethod
#     def _guess_policy_output_index(session, output_names: list[str]) -> int:
#         """
#         Identify which output corresponds to policy logits.
#         Heuristics:
#           1) Name contains 'policy' or 'logit'
#           2) Otherwise, pick the 2D output with the largest last dim (likely vocab size)
#         """
#         outs = session.get_outputs()
#         for i, o in enumerate(outs):
#             nm = o.name.lower()
#             if "policy" in nm or "logit" in nm:
#                 return i

#         best_i, best_score = 0, -1
#         for i, o in enumerate(outs):
#             shape = o.shape
#             rank = len(shape) if shape is not None else 0
#             last = shape[-1] if (shape and shape[-1] is not None) else 0
#             score = (rank == 2) * (100000 + (last or 0))
#             if score > best_score:
#                 best_score = score
#                 best_i = i
#         return best_i

#     # --------- Inference ---------

#     def _tokenize_ids_np(self, board: chess.Board) -> np.ndarray:
#         tokens = fen_to_tokens_fixed77(board.fen())
#         unk_id = self.fen_vocab.get('<UNK>', 0)
#         ids = np.array([[self.fen_vocab.get(t, unk_id) for t in tokens]], dtype=np.int64)  # int64 for ORT
#         return ids

#     def _predict_policy_logits(self, input_ids_np: np.ndarray) -> np.ndarray:
#         outputs = self.ort_session.run(None, {self._onnx_input_name: input_ids_np})
#         policy_logits = outputs[self._policy_out_index]
#         if policy_logits.ndim == 2 and policy_logits.shape[0] == 1:
#             logits = policy_logits[0]
#         else:
#             logits = np.reshape(policy_logits, (-1,))
#         return logits.astype(np.float32)

#     # --------- Public API ---------

#     def get_move(self, board: chess.Board) -> Optional[chess.Move]:
#         legal_uci = [m.uci() for m in board.legal_moves]
#         if not legal_uci:
#             return None

#         input_ids_np = self._tokenize_ids_np(board)
#         logits = self._predict_policy_logits(input_ids_np)

#         legal_ids = np.fromiter(
#             (self.move_vocab[u] for u in legal_uci if u in self.move_vocab),
#             dtype=np.int32
#         )

#         if legal_ids.size == 0:
#             print("Warning: No legal moves found in the model's vocabulary. Picking a random legal move.")
#             return np.random.choice(list(board.legal_moves))

#         move_id = _pick_greedy_legal(logits, legal_ids)
#         best_move_uci = self.move_vocab_inv[move_id]
#         return chess.Move.from_uci(best_move_uci)
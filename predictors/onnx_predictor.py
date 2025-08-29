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


# --- Helpers (same behavior as your original) ---

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
    """Return the token id of the highest-logit legal move."""
    legal_logits = logits[legal_ids]
    return int(legal_ids[int(np.argmax(legal_logits))])


# --- ONNX-only predictor ---

class ONNXMultitaskPredictor(Predictor):
    """
    Predictor that runs an exported ONNX model (policy+value) using onnxruntime.

    Assumes the ONNX model takes an input named 'input_ids' (int64, shape [B, 77]).
    If exported with different names, the class will fall back to the first input and
    heuristically pick the policy output.
    """

    def __init__(
        self,
        model_path: str,
        fen_vocab_path: str,
        move_vocab_path: str,
        providers: Optional[list[str]] = None,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ):
        self.name = "ONNX Multitask"
        # --- Load vocabs ---
        with open(fen_vocab_path, 'r') as f:
            self.fen_vocab = json.load(f)
        with open(move_vocab_path, 'r') as f:
            self.move_vocab = json.load(f)
        self.move_vocab_inv = {v: k for k, v in self.move_vocab.items()}
        self.max_seq_len = 77

        # --- Choose providers (default: CUDA if available, else CPU) ---
        available = ort.get_available_providers()
        if providers is None:
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.device = "cuda"
            else:
                providers = ["CPUExecutionProvider"]
                self.device = "cpu"
        else:
            self.device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        print(f"ONNX providers: {providers}")

        # --- Session options ---
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads is not None:
            so.intra_op_num_threads = int(intra_op_num_threads)
        if inter_op_num_threads is not None:
            so.inter_op_num_threads = int(inter_op_num_threads)

        # --- Create session ---
        self.ort_session = ort.InferenceSession(model_path, sess_options=so, providers=providers)

        # --- Cache IO names ---
        self._onnx_input_name = self._infer_input_name(self.ort_session)
        self._onnx_output_names = [o.name for o in self.ort_session.get_outputs()]
        self._policy_out_index = self._guess_policy_output_index(self.ort_session, self._onnx_output_names)

        print(f"ONNX loaded. Input='{self._onnx_input_name}', Outputs={self._onnx_output_names}, "
              f"PolicyIndex={self._policy_out_index}")

    # --------- IO inference helpers ---------

    @staticmethod
    def _infer_input_name(session) -> str:
        # Prefer a friendly 'input_ids' name if present; else first input.
        for inp in session.get_inputs():
            if inp.name == "input_ids":
                return "input_ids"
        return session.get_inputs()[0].name

    @staticmethod
    def _guess_policy_output_index(session, output_names: list[str]) -> int:
        """
        Identify which output corresponds to policy logits.
        Heuristics:
          1) Name contains 'policy' or 'logit'
          2) Otherwise, pick the 2D output with the largest last dim (likely vocab size)
        """
        outs = session.get_outputs()
        for i, o in enumerate(outs):
            nm = o.name.lower()
            if "policy" in nm or "logit" in nm:
                return i

        best_i, best_score = 0, -1
        for i, o in enumerate(outs):
            shape = o.shape
            rank = len(shape) if shape is not None else 0
            last = shape[-1] if (shape and shape[-1] is not None) else 0
            score = (rank == 2) * (100000 + (last or 0))
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    # --------- Inference ---------

    def _tokenize_ids_np(self, board: chess.Board) -> np.ndarray:
        tokens = fen_to_tokens_fixed77(board.fen())
        unk_id = self.fen_vocab.get('<UNK>', 0)
        ids = np.array([[self.fen_vocab.get(t, unk_id) for t in tokens]], dtype=np.int64)  # int64 for ORT
        return ids

    def _predict_policy_logits(self, input_ids_np: np.ndarray) -> np.ndarray:
        outputs = self.ort_session.run(None, {self._onnx_input_name: input_ids_np})
        policy_logits = outputs[self._policy_out_index]
        if policy_logits.ndim == 2 and policy_logits.shape[0] == 1:
            logits = policy_logits[0]
        else:
            logits = np.reshape(policy_logits, (-1,))
        return logits.astype(np.float32)

    # --------- Public API ---------

    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        legal_uci = [m.uci() for m in board.legal_moves]
        if not legal_uci:
            return None

        input_ids_np = self._tokenize_ids_np(board)
        logits = self._predict_policy_logits(input_ids_np)

        legal_ids = np.fromiter(
            (self.move_vocab[u] for u in legal_uci if u in self.move_vocab),
            dtype=np.int32
        )

        if legal_ids.size == 0:
            print("Warning: No legal moves found in the model's vocabulary. Picking a random legal move.")
            return np.random.choice(list(board.legal_moves))

        move_id = _pick_greedy_legal(logits, legal_ids)
        best_move_uci = self.move_vocab_inv[move_id]
        return chess.Move.from_uci(best_move_uci)
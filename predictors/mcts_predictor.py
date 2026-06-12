# predictors/mcts_predictor.py
"""Batched MCTS (PUCT + virtual loss) on an ONNX policy/value model, CPU-only.

Ported from compact_chess_transformers/encoder_model/search.py (commit
7516623). The strength knob is `nodes` = NN evaluations per move:
  0      -> greedy policy argmax (instant)
  ~6000  -> a few seconds per move on a 4-core CPU (~2.5k evals/s int8)

Cross-move tree reuse is keyed per game_id: the subtree matching the actually
played line carries its visit statistics into the next move's search.

The ONNX graph (export_to_onnx.py in the training repo) outputs:
  policy_logits float [batch, 1968], value_white float [batch] in [-1, 1]
"""
import json
import time
from typing import Dict, List, Optional

import chess
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Board -> 77 token ids, straight from bitboards (no FEN string round-trip)
# ---------------------------------------------------------------------------

class BoardEncoder:
    """Exactly equivalent to fen_to_tokens_fixed77(board.fen()) + vocab lookup."""

    def __init__(self, vocab: dict):
        self.empty_id = vocab['.']
        self.piece_ids = {}
        for i, ch in enumerate("PNBRQK"):
            self.piece_ids[(i + 1, True)] = vocab[ch]
            self.piece_ids[(i + 1, False)] = vocab[ch.lower()]
        self.stm_ids = {True: vocab['w'], False: vocab['b']}
        self.flag_ids = (vocab['0'], vocab['1'])
        self.dash_id = vocab['-']
        self.file_ids = [vocab[c] for c in "abcdefgh"]
        self.rank_ids = [vocab[str(r + 1)] for r in range(8)]
        self.digit_ids = [vocab[d] for d in "0123456789"]

    def encode(self, board: chess.Board) -> np.ndarray:
        ids = np.full(77, self.empty_id, dtype=np.int64)
        for color in (True, False):
            occ = board.occupied_co[color]
            for ptype, bb in ((chess.PAWN, board.pawns), (chess.KNIGHT, board.knights),
                              (chess.BISHOP, board.bishops), (chess.ROOK, board.rooks),
                              (chess.QUEEN, board.queens), (chess.KING, board.kings)):
                m = bb & occ
                pid = self.piece_ids[(ptype, color)]
                while m:
                    sq = (m & -m).bit_length() - 1
                    ids[sq ^ 56] = pid   # token position is FEN order = sq ^ 56
                    m &= m - 1
        ids[64] = self.stm_ids[board.turn]
        cr = board.castling_rights
        flags = self.flag_ids
        ids[65] = flags[bool(cr & chess.BB_H1)]
        ids[66] = flags[bool(cr & chess.BB_A1)]
        ids[67] = flags[bool(cr & chess.BB_H8)]
        ids[68] = flags[bool(cr & chess.BB_A8)]
        if board.ep_square is not None and board.has_legal_en_passant():
            ids[69] = self.file_ids[board.ep_square & 7]
            ids[70] = self.rank_ids[board.ep_square >> 3]
        else:
            ids[69] = self.dash_id
            ids[70] = self.dash_id
        d = self.digit_ids
        h, f = board.halfmove_clock, board.fullmove_number
        ids[71] = d[h // 100 % 10]; ids[72] = d[h // 10 % 10]; ids[73] = d[h % 10]
        ids[74] = d[f // 100 % 10]; ids[75] = d[f // 10 % 10]; ids[76] = d[f % 10]
        return ids


# ---------------------------------------------------------------------------
# MCTS tree
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ('board', 'stm_white', 'terminal_white', 'value_white', 'v_stm',
                 'legal_moves', 'moves', 'move_objs', 'P', 'PU', 'N', 'W', 'Q',
                 'children', 'expanded', 'n_total')

    def __init__(self, board: chess.Board):
        self.board = board
        self.stm_white = board.turn == chess.WHITE
        legal = list(board.legal_moves)
        self.legal_moves = legal
        if not legal:
            self.terminal_white = (-1.0 if board.turn == chess.WHITE else 1.0) \
                if board.is_check() else 0.0
        elif (board.is_insufficient_material() or board.is_seventyfive_moves()
              or board.is_fivefold_repetition()):
            self.terminal_white = 0.0
        else:
            self.terminal_white = None
        self.value_white = 0.0
        self.v_stm = 0.0
        self.moves: List[str] = []
        self.move_objs: List[chess.Move] = []
        self.P = self.PU = self.N = self.W = self.Q = None
        self.children: List[Optional['_Node']] = []
        self.expanded = False
        self.n_total = 0.0

    def expand(self, policy_np, value_white, move_vocab, c_puct, fpu):
        self.value_white = float(value_white)
        self.v_stm = self.value_white if self.stm_white else -self.value_white
        ucis, objs = [], []
        for m in self.legal_moves:
            u = m.uci()
            if u in move_vocab:
                ucis.append(u)
                objs.append(m)
        if not ucis:
            self.terminal_white = self.terminal_white if self.terminal_white is not None else 0.0
            self.expanded = True
            return
        idx = np.fromiter((move_vocab[u] for u in ucis), dtype=np.int64, count=len(ucis))
        logits = policy_np[idx].astype(np.float64)
        logits -= logits.max()
        p = np.exp(logits)
        self.moves, self.move_objs = ucis, objs
        self.P = p / p.sum()
        self.PU = c_puct * self.P
        self.N = np.zeros(len(ucis))
        self.W = np.zeros(len(ucis))
        self.Q = np.full(len(ucis), self.v_stm - fpu)
        self.children = [None] * len(ucis)
        self.expanded = True


def _adopt_subtree(prev_root: _Node, board: chess.Board) -> Optional[_Node]:
    ms = board.move_stack
    if not ms:
        return None
    target_fen = board.fen()
    for k in (1, 2, 3, 4):
        if len(ms) < k:
            break
        node = prev_root
        for mv in ms[-k:]:
            if node is None or not node.expanded or not node.moves:
                node = None
                break
            try:
                i = node.moves.index(mv.uci())
            except ValueError:
                node = None
                break
            node = node.children[i]
        if (node is not None and node.expanded and node.terminal_white is None
                and node.moves and node.board.fen() == target_fen):
            return node
    return None


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

C_PUCT = 1.5
FPU_REDUCTION = 0.3
VIRTUAL_LOSS = 1.0
CACHE_MAX_ENTRIES = 30_000  # ~4KB/entry (fp16 policy) -> ~120MB cap


class MCTSPredictor:
    name = "ONNX-MCTS"

    def __init__(self, model_path, fen_vocab_path, move_vocab_path,
                 batch_size: int = 32, threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), opts,
                                         providers=['CPUExecutionProvider'])
        with open(fen_vocab_path) as f:
            self.fen_vocab = json.load(f)
        with open(move_vocab_path) as f:
            self.move_vocab = json.load(f)
        self.encoder = BoardEncoder(self.fen_vocab)
        self.batch_size = batch_size
        self.eval_cache: Dict[tuple, tuple] = {}
        self.trees: Dict[str, _Node] = {}     # game_id -> last search root

    def drop_game(self, game_id: str):
        self.trees.pop(game_id, None)

    # --- evaluation -------------------------------------------------------

    def _evaluate(self, boards: List[chess.Board]):
        keys = [b._transposition_key() for b in boards]
        missing = [i for i, k in enumerate(keys) if k not in self.eval_cache]
        if missing:
            if len(self.eval_cache) > CACHE_MAX_ENTRIES:
                self.eval_cache.clear()
            ids = np.stack([self.encoder.encode(boards[i]) for i in missing])
            pol, val = self.sess.run(None, {"input_ids": ids})
            for j, i in enumerate(missing):
                self.eval_cache[keys[i]] = (pol[j].astype(np.float16), float(val[j]))
        return [self.eval_cache[k] for k in keys]

    # --- public API -------------------------------------------------------

    def get_move(self, board: chess.Board, nodes: int = 1000,
                 game_id: str = 'default') -> tuple[Optional[chess.Move], dict]:
        """Returns (move, info). nodes=0 -> greedy policy."""
        t0 = time.time()
        legal = list(board.legal_moves)
        if not legal:
            return None, {}
        if nodes <= 0 or len(legal) == 1:
            return self._greedy(board, legal, t0)
        return self._mcts(board, nodes, game_id, t0)

    def _greedy(self, board, legal, t0):
        pol, val = self._evaluate([board])[0]
        pol = pol.astype(np.float32)
        best, best_score = None, -np.inf
        for m in legal:
            mid = self.move_vocab.get(m.uci())
            if mid is not None and pol[mid] > best_score:
                best, best_score = m, pol[mid]
        best = best or legal[0]
        return best, {"win_prob_white": (float(val) + 1) / 2, "nodes": 1,
                      "depth": 0, "time": time.time() - t0}

    def _mcts(self, board, max_evals, game_id, t0):
        root = None
        prev = self.trees.pop(game_id, None)
        if prev is not None:
            root = _adopt_subtree(prev, board)
        if root is None:
            root = _Node(board.copy(stack=False))
            pol, val = self._evaluate([root.board])[0]
            root.expand(pol.astype(np.float32), val, self.move_vocab, C_PUCT, FPU_REDUCTION)
            if not root.moves:
                return self._greedy(board, list(board.legal_moves), t0)
        total_evals = 1
        max_depth = 0

        def select_leaf():
            nonlocal max_depth
            node, path, depth = root, [], 0
            while True:
                if node.terminal_white is not None or not node.expanded:
                    break
                sqrt_n = node.n_total ** 0.5 if node.n_total > 1.0 else 1.0
                i = int(np.argmax(node.Q + node.PU * (sqrt_n / (1.0 + node.N))))
                node.N[i] += VIRTUAL_LOSS
                node.W[i] -= VIRTUAL_LOSS
                node.Q[i] = node.W[i] / node.N[i]
                node.n_total += VIRTUAL_LOSS
                path.append((node, i))
                child = node.children[i]
                if child is None:
                    nb = node.board.copy(stack=False)
                    nb.push(node.move_objs[i])
                    child = _Node(nb)
                    node.children[i] = child
                    depth += 1
                    node = child
                    break
                node = child
                depth += 1
            max_depth = max(max_depth, depth)
            return path, node

        def backup(path, leaf_white_value):
            for parent, i in path:
                contrib = leaf_white_value if parent.stm_white else -leaf_white_value
                parent.N[i] += 1.0 - VIRTUAL_LOSS
                parent.W[i] += contrib + VIRTUAL_LOSS
                parent.Q[i] = parent.W[i] / parent.N[i]
                parent.n_total += 1.0 - VIRTUAL_LOSS

        while total_evals < max_evals:
            if root.n_total > 200:
                bi = int(np.argmax(root.N))
                if root.N[bi] > 100 and root.Q[bi] > 0.99:
                    break
            pending, terminal_backups = {}, []
            for _ in range(min(self.batch_size, max_evals - total_evals + 8)):
                path, leaf = select_leaf()
                if leaf.terminal_white is not None:
                    terminal_backups.append((path, leaf.terminal_white))
                elif leaf.expanded:
                    terminal_backups.append((path, leaf.value_white))
                else:
                    pending.setdefault(id(leaf), (leaf, []))[1].append(path)
            for path, v in terminal_backups:
                backup(path, v)
            if pending:
                nodes_ = [e[0] for e in pending.values()]
                evals = self._evaluate([n.board for n in nodes_])
                total_evals += len(nodes_)
                for node, (pol, vw) in zip(nodes_, evals):
                    node.expand(pol.astype(np.float32), vw, self.move_vocab, C_PUCT, FPU_REDUCTION)
                    for path in pending[id(node)][1]:
                        backup(path, vw)
            elif not terminal_backups:
                break

        best_i = int(np.argmax(root.N))
        self.trees[game_id] = root
        q_stm = float(root.Q[best_i])
        q_white = q_stm if root.stm_white else -q_stm
        info = {"win_prob_white": (q_white + 1) / 2, "nodes": total_evals,
                "depth": max_depth, "time": time.time() - t0,
                "reused_sims": float(root.n_total)}
        return chess.Move.from_uci(root.moves[best_i]), info

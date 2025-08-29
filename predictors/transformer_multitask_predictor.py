import chess
import json
import numpy as np
import torch
import torch.nn as nn

from .base_predictor import Predictor


class FenToPolicyValueTransformer(nn.Module):
    """
    Encoder-only transformer with two heads for multi-task learning:
    1. Policy Head: Predicts the best next move (classification).
    2. Value Head: Predicts the position evaluation (regression).
    """
    def __init__(
        self,
        fen_vocab_size: int,
        uci_vocab_size: int,
        d_model: int = 256,
        n_head: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        max_seq_len: int = 77,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        # --- Shared Body ---
        self.tok_embed = nn.Embedding(fen_vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        
        # --- Heads ---
        self.policy_head = nn.Linear(d_model, uci_vocab_size) # Classification over all moves
        self.value_head = nn.Linear(d_model, 1)              # Scalar regression for eval
        self.tanh = nn.Tanh()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.tok_embed.weight)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, input_ids):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        
        # --- Pass through shared body ---
        x = self.tok_embed(input_ids) + self.pos_embed(positions)
        x = self.encoder(x)
        x = x.mean(dim=1) # Global average pooling
        x = self.norm(self.drop(x))
        
        # --- Pass through heads ---
        policy_logits = self.policy_head(x)
        value_pred = self.tanh(self.value_head(x)).squeeze(-1) # Squeeze to (B,)
        
        return policy_logits, value_pred

# --- Helper functions adapted from your tournament.py ---

def fen_to_tokens_fixed77(fen: str) -> list[str]:
    board, stm, castling, ep, halfmove, fullmove = fen.split()
    expanded = []
    for ch in board:
        if ch == '/': continue
        if ch.isdigit(): expanded.extend(['.'] * int(ch))
        else: expanded.append(ch)
    assert len(expanded) == 64

    toks = expanded + [stm]
    def has(c): return '1' if c in castling else '0'
    toks += [has('K'), has('Q'), has('k'), has('q')]

    if ep != '-': toks += [ep[0], ep[1]]
    else: toks += ['-','-']

    toks += list(f"{int(halfmove):03d}")
    toks += list(f"{int(fullmove):03d}")
    assert len(toks) == 77
    return toks

def _pick_greedy_legal(logits: np.ndarray, legal_ids: np.ndarray) -> int:
    """Return the token id of the highest-logit legal move."""
    legal_logits = logits[legal_ids]
    return int(legal_ids[int(np.argmax(legal_logits))])


class TransformerMultitaskPredictor(Predictor):
    """A predictor that uses your FenToPolicyValueTransformer model."""

    def __init__(self, model_path: str, fen_vocab_path: str, move_vocab_path: str):
        self.name = "Transformer Multitask"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"TorchPredictor using device: {self.device}")

        # 1. Load vocabs
        with open(fen_vocab_path, 'r') as f:
            self.fen_vocab = json.load(f)
        with open(move_vocab_path, 'r') as f:
            self.move_vocab = json.load(f)
        self.move_vocab_inv = {v: k for k, v in self.move_vocab.items()}
        
        self.max_seq_len = 77 # Based on your model's design

        # 2. Instantiate and load the model
        self.model = FenToPolicyValueTransformer(
            fen_vocab_size=len(self.fen_vocab),
            uci_vocab_size=len(self.move_vocab),
            pad_token_id=self.fen_vocab.get('<PAD>', 0)
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("PyTorch model loaded successfully.")

    def get_move(self, board: chess.Board) -> chess.Move:
        legal_uci = [m.uci() for m in board.legal_moves]
        if not legal_uci:
            return None

        # 1. Tokenize the current board state
        tokens = fen_to_tokens_fixed77(board.fen())
        input_ids = torch.tensor(
            [[self.fen_vocab.get(t, self.fen_vocab['<UNK>']) for t in tokens]],
            dtype=torch.long,
            device=self.device
        )

        # 2. Get model prediction
        with torch.no_grad():
            policy_logits, _ = self.model(input_ids)

        logits = policy_logits.squeeze(0).cpu().numpy().astype(np.float32)

        # 3. Get legal move IDs from vocab
        legal_ids = np.fromiter(
            (self.move_vocab[u] for u in legal_uci if u in self.move_vocab),
            dtype=np.int32
        )
        
        if legal_ids.size == 0:
            print("Warning: No legal moves found in the model's vocabulary. Picking a random legal move.")
            return np.random.choice(list(board.legal_moves))

        # 4. Pick the best legal move (greedy approach)
        move_id = _pick_greedy_legal(logits, legal_ids)
        best_move_uci = self.move_vocab_inv[move_id]

        return chess.Move.from_uci(best_move_uci)
import torch
import torch.nn as nn
from data import PAD, SEQ_LEN, N_NUM, CATS


class TransFraud(nn.Module):
    """Small Transformer encoder over a user's recent transaction history.
    Each transaction -> (numeric projection + category embedding + position).
    Self-attention lets the last transaction be compared with the user's past behaviour."""

    def __init__(self, n_cats=len(CATS), d=32, heads=4, layers=2, L=SEQ_LEN):
        super().__init__()
        self.num_proj = nn.Linear(N_NUM, d)
        self.cat_emb = nn.Embedding(n_cats + 1, d, padding_idx=PAD)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))

    def _embed(self, xnum, xcat):
        return self.num_proj(xnum) + self.cat_emb(xcat) + self.pos

    def forward(self, xnum, xcat):
        pad_mask = xcat == PAD
        h = self.encoder(self._embed(xnum, xcat), src_key_padding_mask=pad_mask)
        return self.head(h[:, -1]).squeeze(-1)   # logit for the latest transaction

    @torch.no_grad()
    def attention_to_last(self, xnum, xcat):
        """For visualisation: how much the LAST transaction attends to each of the L positions
        (averaged over heads and layers). Returns a (batch, L) tensor that sums to ~1 per row."""
        pad_mask = xcat == PAD
        h = self._embed(xnum, xcat)
        maps = []
        for layer in self.encoder.layers:
            a, w = layer.self_attn(h, h, h, key_padding_mask=pad_mask,
                                   need_weights=True, average_attn_weights=True)
            maps.append(w[:, -1, :])
            h = layer.norm1(h + layer.dropout1(a))
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm2(h + layer.dropout2(ff))
        return torch.stack(maps).mean(0)


class LSTMBaseline(nn.Module):
    """Comparison model: an LSTM over the same 10-transaction window (no attention)."""

    def __init__(self, n_cats=len(CATS), d=32):
        super().__init__()
        self.num_proj = nn.Linear(N_NUM, d)
        self.cat_emb = nn.Embedding(n_cats + 1, d, padding_idx=PAD)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, xnum, xcat):
        h, _ = self.lstm(self.num_proj(xnum) + self.cat_emb(xcat))
        return self.head(h[:, -1]).squeeze(-1)

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
import math


class DecoderTransformerUnDirected(nn.Module):
    """
    Set Transformer backbone (Lee et al., 2019) + bilinear edge head.
    Self-attention contextualizes all nodes globally before pairwise edge prediction.
    DeepSet is omitted here since the transformer already captures global context.
    """

    def __init__(self, pe_dim, num_node_features, num_edge_features, num_heads,
                 num_layers, hidden_dim, dropout):
        super().__init__()
        self.pe_dim = pe_dim
        self.num_edge_features = num_edge_features + 1

        # Node contextualization via transformer
        self.in_proj = nn.Linear(pe_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN for stability (same convention as DiT)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                  enable_nested_tensor=False)
        self.out_proj = nn.Linear(hidden_dim, pe_dim)

        # Bilinear edge head (no DeepSet — transformer already has global context)
        self.attn_proj = nn.Linear(pe_dim, 2 * self.num_edge_features * pe_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.node_out_proj = nn.Linear(pe_dim, num_node_features) if num_node_features > 0 else nn.Identity()

    def forward(self, x, batch, q=None):
        z_dense, mask = to_dense_batch(x, batch)                                # (bs, n, d), (bs, n)

        # Contextualize nodes; ~mask marks padding positions to ignore
        h = self.in_proj(z_dense)
        h = self.transformer(h, src_key_padding_mask=~mask)                     # (bs, n, hidden_dim)
        h = self.out_proj(h)                                                    # (bs, n, pe_dim)

        x_ctx = h[mask]                                                         # (n_total, pe_dim)
        x_hat = self.node_out_proj(x_ctx)                                       # (n_total, nf)

        # Bilinear edge scores
        k, q_vec = self.attn_proj(x_ctx).split(self.num_edge_features * self.pe_dim, dim=-1)
        k = self.attn_dropout(k)
        q_vec = self.attn_dropout(q_vec)

        k, _ = to_dense_batch(k, batch)                                         # (bs, n, e*d)
        q_vec, _ = to_dense_batch(q_vec, batch)
        scale = 1 / math.sqrt(self.num_edge_features * self.pe_dim)
        bs, n, _ = k.shape

        k = k.reshape(bs, n, self.num_edge_features, -1).transpose(1, 2)       # (bs, e, n, d)
        q_vec = q_vec.reshape(bs, n, self.num_edge_features, -1).transpose(1, 2)
        z_tilde = torch.einsum("bend,bemd->benm", q_vec, k) * scale            # (bs, e, n, n)
        z_tilde = (z_tilde + z_tilde.transpose(-2, -1)) / 2                    # symmetrize

        upper = torch.triu(z_tilde, diagonal=1)
        e_hat = upper + upper.transpose(-1, -2)

        edge_mask = self._build_edge_mask(mask)
        return e_hat.transpose(-1, 1), x_hat, edge_mask, mask                  # (bs, n, n, e)

    def _build_edge_mask(self, mask):
        b, n = mask.shape
        edge_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        diag = torch.eye(n, dtype=torch.bool, device=mask.device).unsqueeze(0).expand(b, -1, -1)
        return edge_mask & ~diag



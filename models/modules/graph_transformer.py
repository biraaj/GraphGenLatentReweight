import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, to_dense_batch


class GraphTransformerAggregator(nn.Module):
    """Lightweight alternative to GIN (models/modules/gin.py): dense multi-head
    self-attention over all node pairs, biased by RRWP (Ma et al., ICML 2023, "GRIT")
    instead of a binary edge/no-edge bias, which caused posterior collapse on sparse
    graphs. RRWP is automorphism-invariant, so this complements rather than replaces
    orbit modulation/rGIN (see EncoderGINVAE)."""

    def __init__(
        self,
        num_layers: int,
        in_dim: int,
        embed_dim: int,
        pe_dim: int,
        num_heads: int = 4,
        dropout: float = 0.,
        rrwp_steps: int = 16,
        rrwp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.rrwp_steps = rrwp_steps
        self.dropout = dropout
        self.in_proj = nn.Linear(in_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN, same convention as DecoderTransformerUnDirected / DiT.
        )
        # Final LayerNorm bounds the otherwise-unnormalized Pre-LN residual stream's scale.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False
        )

        # RRWP -> per-head attention bias, shared across all layers. rrwp_hidden=None is
        # a single cheap Linear; a nonzero hidden width materializes a dense (bs,n,n,h)
        # tensor that can OOM at large n/batch_size.
        if rrwp_hidden is None:
            self.rrwp_proj = nn.Linear(rrwp_steps, num_heads)
        else:
            self.rrwp_proj = nn.Sequential(
                nn.Linear(rrwp_steps, rrwp_hidden),
                nn.GELU(),
                nn.Linear(rrwp_hidden, num_heads),
            )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * pe_dim),
        )

    def _rrwp_bias(self, edge_index: torch.Tensor, batch: torch.Tensor, bs: int, n: int) -> torch.Tensor:
        # Fixed function of graph structure; no gradient needed through the adjacency powers.
        with torch.no_grad():
            adj = to_dense_adj(edge_index=edge_index, batch=batch, max_num_nodes=n)  # (bs, n, n)
            deg = adj.sum(-1, keepdim=True).clamp_min(1.0)                          # (bs, n, 1)
            transition = adj / deg                                                  # (bs, n, n), M = D^-1 A
            eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(bs, -1, -1)
            powers = [eye]
            power = eye
            for _ in range(self.rrwp_steps - 1):
                power = power @ transition
                powers.append(power)
            rrwp = torch.stack(powers, dim=-1)                                      # (bs, n, n, rrwp_steps)

        bias = self.rrwp_proj(rrwp)                          # (bs, n, n, heads)
        bias = bias.permute(0, 3, 1, 2)                      # (bs, heads, n, n)
        return bias.reshape(bs * self.num_heads, n, n)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_dense, mask = to_dense_batch(x, batch)             # (bs, n, in_dim), (bs, n)
        bs, n, _ = x_dense.shape

        attn_bias = self._rrwp_bias(edge_index, batch, bs, n)  # (bs*heads, n, n)
        h = self.in_proj(x_dense)

        # PyTorch's eval-mode fast path silently NaNs when combining a float mask (RRWP
        # bias) with src_key_padding_mask; forcing train-mode internals works around it
        # (safe since dropout=0 everywhere, so train/eval behave identically otherwise).
        was_training = self.transformer.training
        if self.dropout == 0.:
            self.transformer.train()
        h = self.transformer(h, mask=attn_bias, src_key_padding_mask=~mask)
        if self.dropout == 0.:
            self.transformer.train(was_training)

        h = self.mlp(h)                                        # (bs, n, 2*pe_dim)
        return h[mask]                                         # (n_total, 2*pe_dim)

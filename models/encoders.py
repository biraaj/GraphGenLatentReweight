import torch
from torch import nn
from typing import Any
from .modules import GIN
from .transformer import get_pos_embedding
from loguru import logger


class NodeFeatureEmbedder(nn.Module):
    def __init__(self, num_node_types, num_node_features, embed_dim):
        super().__init__()
        self.num_node_types = num_node_types
        self.num_node_features = num_node_features

        if num_node_types > 0 and num_node_features > 1:
            self.mode = "mixed"
            self.type_embed = nn.Embedding(num_node_types, embed_dim)
            self.feat_proj = nn.Linear(num_node_features - 1, embed_dim)
        elif num_node_types > 0:
            self.mode = "types"
            self.embed = nn.Embedding(num_node_types, embed_dim)
        else:
            self.mode = "identity"
            self.embed = nn.Identity()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        legacy_weight_key = f"{prefix}weight"
        legacy_bias_key = f"{prefix}bias"
        remapped_keys: list[str] = []

        if legacy_weight_key in state_dict:
            legacy_weight = state_dict[legacy_weight_key]
            legacy_bias = state_dict.get(legacy_bias_key)

            if self.mode == "types":
                current_weight_key = f"{prefix}embed.weight"
                if (
                    legacy_weight.shape == self.embed.weight.shape
                    and legacy_bias is None
                    and current_weight_key not in state_dict
                ):
                    state_dict[current_weight_key] = state_dict.pop(legacy_weight_key)
                    remapped_keys.append(current_weight_key)
                elif (
                    legacy_weight.dim() == 2
                    and legacy_weight.size(1) == 1
                    and current_weight_key not in state_dict
                ):
                    type_ids = torch.arange(
                        self.num_node_types,
                        device=legacy_weight.device,
                        dtype=legacy_weight.dtype,
                    )  # (num_node_types,)
                    type_embed = type_ids.unsqueeze(1) * legacy_weight[:, 0].unsqueeze(0)  # (num_node_types, embed_dim)
                    if legacy_bias is not None:
                        type_embed = type_embed + legacy_bias.unsqueeze(0)  # (num_node_types, embed_dim)
                        state_dict.pop(legacy_bias_key)
                    state_dict[current_weight_key] = type_embed
                    state_dict.pop(legacy_weight_key)
                    remapped_keys.append(current_weight_key)

            elif (
                self.mode == "mixed"
                and legacy_weight.dim() == 2
                and legacy_weight.size(1) == self.num_node_features
            ):
                type_weight_key = f"{prefix}type_embed.weight"
                feat_weight_key = f"{prefix}feat_proj.weight"
                feat_bias_key = f"{prefix}feat_proj.bias"
                if type_weight_key not in state_dict and feat_weight_key not in state_dict:
                    type_ids = torch.arange(
                        self.num_node_types,
                        device=legacy_weight.device,
                        dtype=legacy_weight.dtype,
                    )  # (num_node_types,)
                    type_embed = type_ids.unsqueeze(1) * legacy_weight[:, 0].unsqueeze(0)  # (num_node_types, embed_dim)
                    state_dict[type_weight_key] = type_embed
                    state_dict[feat_weight_key] = legacy_weight[:, 1:]  # (embed_dim, num_node_features - 1)
                    remapped_keys.extend([type_weight_key, feat_weight_key])
                    if legacy_bias is not None and feat_bias_key not in state_dict:
                        state_dict[feat_bias_key] = legacy_bias
                        remapped_keys.append(feat_bias_key)
                        state_dict.pop(legacy_bias_key)
                    state_dict.pop(legacy_weight_key)

        if remapped_keys:
            logger.info(
                f"Remapped legacy node feature embedder checkpoint keys under `{prefix}` to {remapped_keys}"
            )

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    def forward(self, x):
        if self.mode == "identity":
            return x.float()

        if self.mode == "types":
            x_type = x if x.dim() == 1 else x[:, 0]
            return self.embed(x_type.long())

        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_type = x[:, 0].long()
        x_feat = x[:, 1:].float()
        return self.type_embed(x_type) + self.feat_proj(x_feat)


class PaddedEigenEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, bias):
        super().__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(2, 2 * embed_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 2, embed_dim, bias=bias),
            torch.nn.ReLU(),
        )

    def forward(self, eigvecs, eigvals):
        x = torch.stack((eigvecs, eigvals), 2)
        empty_mask = torch.isnan(x)
        x[empty_mask] = 0
        return self.ffn(x)


def nan_to_zero(mat):
    # Out-of-place: in-place mutation would break autograd for other consumers of this shared view.
    return torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)


class EncoderGINVAE(nn.Module):
    """VGAE-style encoder: LPE eigenvectors projected to an embedding, then a GIN.
    Opt-in flags (all off by default): sign_flip_aug (sign ambiguity augmentation),
    modulation/modulation_mode (orbit-breaking noise, multiplicative or additive),
    rni_dim (rGIN random node init), spectral_phi (per-eigenvector FFN instead of a
    flat linear projection), node_id_pe, modulation_always_on."""

    def __init__(self, num_node_types, num_node_features, num_edge_features,
                 global_cfg, rho_cfg, dropout):
        super().__init__()
        embed_dim = rho_cfg.embed_dim
        self.num_vecs = global_cfg.num_vecs
        self.modulation = global_cfg.modulation
        self.modulation_mode = getattr(global_cfg, "modulation_mode", "multiplicative")
        if self.modulation_mode not in ("multiplicative", "additive"):
            raise ValueError(f"Unknown modulation_mode: {self.modulation_mode!r}")
        self.pe_type = getattr(global_cfg, "pe_type", "lpe")
        self.sign_flip_aug = getattr(global_cfg, "sign_flip_aug", False)
        self.spectral_phi = getattr(global_cfg, "spectral_phi", False)
        self.rni_dim = getattr(global_cfg, "rni_dim", 0)
        self.node_id_pe = getattr(global_cfg, "node_id_pe", False)
        # modulation_always_on: drop the self.training gate so modulation fires in eval too; off by default.
        self.modulation_always_on = getattr(global_cfg, "modulation_always_on", False)

        self.node_features_embed = NodeFeatureEmbedder(
            num_node_types=num_node_types,
            num_node_features=num_node_features,
            embed_dim=embed_dim,
        )
        if self.rni_dim > 0:
            self.rni_proj = nn.Linear(self.rni_dim, embed_dim)
        if self.spectral_phi:
            # Original LG-VAE-style per-eigenvector FFN on (v_i, lambda_i), summed over i.
            self.phi = PaddedEigenEmbedding(embed_dim, bias=True)
            self.eig_eps = nn.Parameter(1e-12 * torch.arange(self.num_vecs).unsqueeze(0))
        else:
            # Simple linear projection of all eigenvectors at once — replaces the
            # per-vector FFN (phi) used in LG-VAE
            self.lpe_proj = nn.Linear(global_cfg.num_vecs, embed_dim)
        self.edge_features_embed = (
            nn.Linear(num_edge_features, embed_dim) if num_edge_features > 0 else nn.Identity()
        )
        self.rho = GIN(False, rho_cfg.num_layers, embed_dim, embed_dim, global_cfg.pe_dim, dropout, rho_cfg.norm)

    def _edge_embed(self, data):
        return self.edge_features_embed(data.edge_attr.float())

    def forward(self, data):
        eigvecs = data.eigvecs[:, :self.num_vecs]                           # (n_sum, num_vecs)
        eigvecs = nan_to_zero(eigvecs)

        # Sign-flip augmentation: eigenvectors are sign-ambiguous; skipped for RRWP (non-negative).
        if self.sign_flip_aug and self.training and self.pe_type not in ("rrwp", "rrwp_full"):
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)

        # Orbit modulation: opt-in per dataset via modulation > 0 (0 = no-op).
        if self.modulation > 0. and (self.training or self.modulation_always_on):
            m = (data.orbits != 0).to(eigvecs.dtype)                        # (n_sum, 1) in {0,1}
            if self.modulation_mode == "additive":
                # Additive: independent per-element noise breaks colinearity directly.
                noise = torch.randn_like(eigvecs)                           # (n_sum, num_vecs)
                eigvecs = eigvecs + self.modulation * noise * m
            else:
                g = torch.randn(eigvecs.size(0), 1, device=eigvecs.device, dtype=eigvecs.dtype)
                factor = 1 + self.modulation * g * m
                eigvecs = factor * eigvecs

        x_embed = self.node_features_embed(data.x)                         # (n_sum, d)
        if self.spectral_phi:
            eigvals = data.eigvals[:, :self.num_vecs] + self.eig_eps[:, :eigvecs.size(1)]
            lpe = self.phi(eigvecs, eigvals).sum(1)                        # (n_sum, d) — sum over eigvec index
        else:
            lpe = self.lpe_proj(eigvecs)                                   # flat LPE projection
        x_embed = x_embed + lpe

        # Deterministic node-index positional encoding (opt-in, off by default) --
        # breaks permutation equivariance; valid since exact-match reconstruction is
        # order-dependent (see Murphy et al., ICML 2019).
        if self.node_id_pe:
            batch = getattr(data, "batch", None)
            n_total = x_embed.size(0)
            if batch is not None:
                counts = torch.bincount(batch, minlength=int(batch.max().item()) + 1)
                ptr = torch.zeros(counts.numel() + 1, dtype=torch.long, device=batch.device)
                ptr[1:] = counts.cumsum(0)
                local_idx = torch.arange(n_total, device=x_embed.device) - ptr[batch]
            else:
                local_idx = torch.arange(n_total, device=x_embed.device)
            x_embed = x_embed + get_pos_embedding(local_idx.float(), x_embed.size(-1))

        # Random Node Initialization (rGIN): resampled every call, train and eval alike.
        if self.rni_dim > 0:
            r = torch.randn(x_embed.size(0), self.rni_dim, device=x_embed.device, dtype=x_embed.dtype)
            x_embed = x_embed + self.rni_proj(r)

        e_embed = self._edge_embed(data)
        return self.rho(x_embed, data.edge_index, e_embed,
                        getattr(data, "batch", None))                       # (n_sum, 2*pe_dim)



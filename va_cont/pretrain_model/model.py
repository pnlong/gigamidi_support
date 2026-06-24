"""GPT-style causal transformer for bar-level continuous valence/arousal regression.

Two variants controlled by va_conditioning:
  Model A (va_conditioning=False): each bar conditioned on all prior bar latents only.
  Model B (va_conditioning=True):  each bar additionally conditioned on the previous
      bar's (valence, arousal) prediction (teacher-forced during training;
      sequential AR decoding at inference).
"""

import math
import torch
import torch.nn as nn


class CausalVATransformer(nn.Module):
    """
    GPT-style causal transformer predicting (valence, arousal) per bar.

    Args:
        latent_dim:      MuseTok latent dimension (default 128).
        d_model:         Transformer hidden dimension (default 128).
        n_heads:         Attention heads (default 4).
        n_layers:        Transformer layers (default 2).
        d_ff:            FFN intermediate dimension (default 256).
        dropout:         Dropout rate (default 0.1).
        max_len:         Maximum sequence length for positional encoding (default 512).
        va_conditioning: If True (Model B), concat prev_va to each bar input.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        va_conditioning: bool = False,
    ):
        super().__init__()
        self.va_conditioning = va_conditioning
        self.d_model = d_model
        input_dim = latent_dim + (2 if va_conditioning else 0)

        # Learnable null VA token — used for bar 0 and unannotated predecessors (Model B).
        # Always registered so checkpoints are consistent regardless of va_conditioning.
        self.null_va = nn.Parameter(torch.zeros(2))

        # Input projection (handles both input_dim == d_model and input_dim != d_model)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Sinusoidal positional encoding (fixed, not learned)
        self.register_buffer("pos_enc", self._make_sinusoidal(max_len, d_model))

        # Pre-LN transformer encoder (norm_first=True is more stable; requires torch >= 1.11)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Per-position regression head
        self.output_head = nn.Linear(d_model, 2)

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _make_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        """Standard sinusoidal positional encoding, shape (max_len, d_model)."""
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        return pe  # (max_len, d_model)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular additive -inf mask so position i attends only to 0..i."""
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def _build_prev_va(
        self, va_targets: torch.Tensor, label_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Build the teacher-forced prev_va tensor for Model B.

        prev_va[:, 0]  = null_va  (no predecessor for bar 0)
        prev_va[:, t]  = va_targets[:, t-1]  if label_mask[:, t-1]
                       = null_va              otherwise

        Args:
            va_targets:  (B, T, 2) — ground-truth VA; 0.0 for unannotated bars
            label_mask:  (B, T) bool — True for annotated bars

        Returns:
            (B, T, 2)
        """
        B, T, _ = va_targets.shape
        null = self.null_va.detach().view(1, 1, 2).expand(B, 1, 2)  # (B, 1, 2)

        if T == 1:
            return null.clone()

        # Positions 1..T-1: use va_targets[t-1] where annotated, else null_va
        prior_targets = va_targets[:, :-1, :]                       # (B, T-1, 2)
        prior_mask = label_mask[:, :-1].unsqueeze(-1).expand_as(prior_targets)
        null_fill = self.null_va.detach().view(1, 1, 2).expand_as(prior_targets)
        prior = torch.where(prior_mask, prior_targets, null_fill)   # (B, T-1, 2)

        return torch.cat([null.clone(), prior], dim=1)               # (B, T, 2)

    # ------------------------------------------------------------------ #
    #  Forward (training / teacher-forced evaluation)                     #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        latents: torch.Tensor,
        padding_mask: torch.BoolTensor = None,
        va_targets: torch.Tensor = None,
        label_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """
        Args:
            latents:      (B, T, latent_dim)
            padding_mask: (B, T) bool — True = padded position (PyTorch convention).
                          Pass None for unbatched / no-padding inference.
            va_targets:   (B, T, 2) — required for Model B (teacher forcing).
                          Unannotated bars should be 0.0; use label_mask to mark them.
            label_mask:   (B, T) bool — True = annotated bar. Required for Model B.

        Returns:
            (B, T, 2) per-bar (valence, arousal) predictions.
        """
        B, T, _ = latents.shape

        if self.va_conditioning:
            assert va_targets is not None and label_mask is not None, (
                "va_targets and label_mask are required for Model B (va_conditioning=True)"
            )
            prev_va = self._build_prev_va(va_targets, label_mask)  # (B, T, 2)
            x = torch.cat([latents, prev_va], dim=-1)              # (B, T, 130)
        else:
            x = latents                                             # (B, T, 128)

        # Project to d_model and add positional encoding
        x = self.input_proj(x) + self.pos_enc[:T].unsqueeze(0)    # (B, T, d_model)

        causal_mask = self._make_causal_mask(T, x.device)
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )

        return self.output_head(x)  # (B, T, 2)

    # ------------------------------------------------------------------ #
    #  Sequential AR inference (Model B only)                             #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def infer_sequential(
        self, latents: torch.Tensor, differential: bool = False
    ) -> torch.Tensor:
        """
        True autoregressive inference for Model B: each bar's prediction is fed
        back as prev_va for the next bar. No teacher forcing.

        Args:
            latents:      (1, T, latent_dim) — single song, no padding.
            differential: If True, the model was trained in differential mode and its
                          raw output is ΔV/ΔA per bar. In that case prev_va is built
                          from the running integrated absolute VA (matching training,
                          where _build_prev_va used absolute va_targets). The returned
                          tensor still contains raw model outputs (deltas); callers must
                          apply cumsum to obtain absolute values.

        Returns:
            (T, 2) — raw model output per bar (absolute if differential=False,
                     ΔV/ΔA if differential=True).

        Complexity: O(T²) — KV-cache can be added later for long songs.
        """
        assert self.va_conditioning, "infer_sequential is only for Model B"
        assert latents.shape[0] == 1, "infer_sequential expects a single song (B=1)"

        T = latents.shape[1]
        predictions: list[torch.Tensor] = []   # raw model output per bar
        # running_abs tracks the integrated absolute VA for prev_va feedback.
        # Initialised to null_va so bar 0 receives null_va as its predecessor.
        running_abs: list[torch.Tensor] = [self.null_va.detach().clone()]  # (2,) per step

        for t in range(T):
            # prev_va[i] = integrated absolute VA for bar i's predecessor
            # = running_abs[0] (null_va) for bar 0, running_abs[t] for bar t
            prev_list = [v.view(1, 2) for v in running_abs]  # t+1 entries
            prev_va = torch.stack(prev_list, dim=1)  # (1, t+1, 2)

            lat_prefix = latents[:, : t + 1, :]   # (1, t+1, latent_dim)
            x = torch.cat([lat_prefix, prev_va], dim=-1)  # (1, t+1, latent_dim+2)
            x = self.input_proj(x) + self.pos_enc[: t + 1].unsqueeze(0)

            causal_mask = self._make_causal_mask(t + 1, x.device)
            out = self.transformer(x, mask=causal_mask, is_causal=True)  # (1, t+1, d_model)

            pred_t = self.output_head(out[0, t, :])  # (2,) — raw prediction for bar t
            predictions.append(pred_t)

            # Update integrated absolute VA for next bar's prev_va
            if differential:
                new_abs = running_abs[-1] + pred_t   # accumulate delta
            else:
                new_abs = pred_t                      # already absolute
            running_abs.append(new_abs.detach())

        return torch.stack(predictions, dim=0)  # (T, 2)

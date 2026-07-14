"""Soft-binned VA targets on [-1, 1] for classification-style training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_bin_centers(n_bins: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Midpoints of n_bins equal-width bins on [-1, 1]. Shape (n_bins,)."""
    i = torch.arange(n_bins, device=device, dtype=dtype)
    return -1.0 + (i + 0.5) * (2.0 / n_bins)


def default_bin_sigma(n_bins: int) -> float:
    """Default Gaussian width: one bin width in VA space."""
    return 2.0 / n_bins


def va_to_soft_bin_targets(
    va: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian soft labels over bins for valence and arousal.

    Args:
        va:           (N, 2) continuous targets in [-1, 1]
        bin_centers:  (n_bins,)
        sigma:        Gaussian std in VA units

    Returns:
        soft_v, soft_a: each (N, n_bins), rows sum to 1
    """
    centers = bin_centers.view(1, -1)
    v = va[:, 0:1].clamp(-1.0, 1.0)
    a = va[:, 1:2].clamp(-1.0, 1.0)
    soft_v = torch.exp(-0.5 * ((v - centers) / sigma) ** 2)
    soft_a = torch.exp(-0.5 * ((a - centers) / sigma) ** 2)
    soft_v = soft_v / soft_v.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    soft_a = soft_a / soft_a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return soft_v, soft_a


def binned_va_kl_loss(
    logits: torch.Tensor,
    soft_v: torch.Tensor,
    soft_a: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    """KL(soft_target || softmax(logits)) averaged over valence and arousal."""
    v_logits = logits[:, :n_bins]
    a_logits = logits[:, n_bins:]
    kl_v = F.kl_div(F.log_softmax(v_logits, dim=-1), soft_v, reduction="batchmean")
    kl_a = F.kl_div(F.log_softmax(a_logits, dim=-1), soft_a, reduction="batchmean")
    return 0.5 * (kl_v + kl_a)


def decode_binned_logits(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    """Expected VA under per-dimension softmax. (..., 2*n_bins) -> (..., 2)."""
    v_logits = logits[..., :n_bins]
    a_logits = logits[..., n_bins:]
    pv = F.softmax(v_logits, dim=-1)
    pa = F.softmax(a_logits, dim=-1)
    v = (pv * bin_centers).sum(dim=-1)
    a = (pa * bin_centers).sum(dim=-1)
    return torch.stack([v, a], dim=-1)

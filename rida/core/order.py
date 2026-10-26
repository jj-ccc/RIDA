"""Pairwise order of the similarities to the image anchors.

For a pixel with weak-view similarities ``s`` and strong-view similarities ``s'`` to ``K`` image anchors, the
pairwise gaps ``d_ij = s_i - s_j`` define the order the method reads.  The trusted pairs of a pixel
are the pairs whose weak-view gap exceeds the ``q``-quantile ``delta`` of all gaps in the batch.
One set of trusted pairs serves both order terms (``rank_terms``):

* ``L_rc`` -- hard order matching: the weak-view order ``1[d_ij > 0]`` is the target of a logistic
  loss on the strong-view gap scaled by ``delta``;
* rank stability -- the fraction of trusted pairs whose order survives the strong view; it weights
  the pseudo-label of every pixel (``L_rs``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["pairwise_order", "rank_terms"]


def pairwise_order(sim_weak: torch.Tensor, sim_strong: torch.Tensor, q: float):
    """Pairwise gaps of both views and the quantile gate.

    Returns:
        ``d_w`` and ``d_s`` of shape ``(M, K, K)``, the boolean gate, the scalar ``delta`` and the
        off-diagonal mask.
    """
    _, _, _, k = sim_weak.shape
    s_w = sim_weak.reshape(-1, k).detach()
    s_s = sim_strong.reshape(-1, k)
    d_w = s_w.unsqueeze(2) - s_w.unsqueeze(1)
    d_s = s_s.unsqueeze(2) - s_s.unsqueeze(1)
    offdiag = ~torch.eye(k, dtype=torch.bool, device=sim_weak.device).unsqueeze(0)
    gaps = d_w.abs()[:, offdiag.squeeze(0)].reshape(-1)
    if gaps.numel() > 1_000_000:
        gaps = gaps[:: (gaps.numel() + 999_999) // 1_000_000]
    delta = torch.quantile(gaps, float(q))
    gate = offdiag & (d_w.abs() > delta)
    return d_w, d_s, gate, delta, offdiag


def rank_terms(sim_weak: torch.Tensor, sim_strong: torch.Tensor, q: float):
    """``L_rc`` and the per-pixel rank stability from one set of trusted pairs.

    A pixel keeps the stability ``s`` (fraction of its trusted pairs with an unchanged order) as its
    weight when ``s >= 0.8`` and at least 10% of its pairs are trusted.

    Returns:
        The scalar ``L_rc``, the stability weights ``(B, H, W)`` and the fraction of weighted pixels.
    """
    b, h, w, _ = sim_weak.shape
    d_w, d_s, gate, delta, offdiag = pairwise_order(sim_weak, sim_strong, q)
    gate_f = gate.float()
    target = (d_w > 0).to(d_s.dtype)
    per_pair = F.binary_cross_entropy_with_logits(d_s / delta.clamp_min(1e-4), target, reduction="none")
    loss = (gate_f * per_pair).sum() / gate_f.sum().clamp_min(1.0)
    with torch.no_grad():
        selected = gate_f.sum(dim=(1, 2))
        agree = ((torch.sign(d_w) == torch.sign(d_s)) & gate).float().sum(dim=(1, 2))
        stability = agree / selected.clamp_min(1.0)
        pair_frac = selected / float(offdiag.float().sum().clamp_min(1.0).item())
        keep = (selected > 0) & (stability >= 0.8) & (pair_frac >= 0.1)
        weights = torch.where(keep, stability, torch.zeros_like(stability))
    return loss, weights.reshape(b, h, w).clamp_(0.0, 1.0), keep.float().mean()

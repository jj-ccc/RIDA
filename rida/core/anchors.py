"""Image anchors and the similarities of every pixel to them.

A pixel is compared, by cosine similarity, with a small set of tokens of its own image, the
*image anchors*, chosen per image by

* ``ortho_anchors``  -- seeded Gram-Schmidt selection: the first anchor is the token closest to the
  mean feature direction, every further anchor is the token with the largest residual after
  projecting out the anchors already chosen.

``seeded_ortho`` applies the same rule to a set of feature rows; the cross-domain signature bank
uses it to initialise the class anchors of ``L_rm`` (``signatures.py``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "anchor_similarities",
    "ortho_anchors",
    "seeded_ortho",
    "row_rank",
    "rank_correlation",
]


def anchor_similarities(fea: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of every pixel to the image anchors ``indices``.

    Args:
        fea: ``(B, C, H, W)`` feature map.
        indices: ``(B, K)`` flattened token indices of the image anchors.

    Returns:
        ``(B, H, W, K)`` similarities.
    """
    b, c, h, w = fea.size()
    fea_2d = F.normalize(fea.view(b, c, -1), p=2, dim=1)
    indices = indices.to(device=fea.device, dtype=torch.long)
    gather_index = indices.unsqueeze(1).expand(-1, c, -1)
    anchors = torch.gather(fea_2d, dim=-1, index=gather_index)
    sim = torch.bmm(fea_2d.transpose(1, 2), anchors)
    return sim.view(b, h, w, indices.shape[1])


def ortho_anchors(fea: torch.Tensor, k: int) -> torch.Tensor:
    """``(B, K)`` seeded orthogonal image anchors of every image (token indices), selected batched over the images."""
    b, c, h, w = fea.size()
    f = F.normalize(fea.detach().reshape(b, c, -1), p=2, dim=1).transpose(1, 2)
    n = f.shape[1]
    ar = torch.arange(b, device=f.device)
    typ = torch.bmm(f, F.normalize(f.mean(1), dim=1)[:, :, None])[:, :, 0]
    chosen = [typ.argmax(dim=1)]
    taken = torch.zeros(b, n, dtype=torch.bool, device=f.device)
    taken[ar, chosen[0]] = True
    res = f.clone()
    for _ in range(1, min(int(k), n)):
        v = res[ar, chosen[-1]]
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        res = res - torch.bmm(res, v[:, :, None]) * v[:, None, :]
        r = (res * res).sum(2).masked_fill(taken, -1.0)
        chosen.append(r.argmax(dim=1))
        taken[ar, chosen[-1]] = True
    return torch.stack(chosen, dim=1)


@torch.no_grad()
def seeded_ortho(x: torch.Tensor, k: int, q: float = 0.7) -> torch.Tensor:
    """``k`` representative and mutually orthogonal unit rows of ``x`` ``(N, C)``."""
    mu = F.normalize(x.mean(0), dim=0)
    typ = x @ mu
    pool = x[typ >= torch.quantile(typ.float(), float(q))] if q > 0 and x.shape[0] > k else x
    if pool.shape[0] <= k:
        return pool.clone()
    chosen = [int((pool @ mu).argmax())]
    res = pool.clone()
    for _ in range(1, k):
        qv = res[chosen[-1]] / res[chosen[-1]].norm().clamp_min(1e-12)
        res = res - (res @ qv)[:, None] * qv[None, :]
        r = (res * res).sum(1)
        r[chosen] = -1.0
        chosen.append(int(r.argmax()))
    return pool[chosen]


def row_rank(x: torch.Tensor) -> torch.Tensor:
    """Ascending rank of every row."""
    return x.argsort(dim=1).argsort(dim=1).to(x.dtype)


def rank_correlation(rank_a: torch.Tensor, rank_b: torch.Tensor) -> torch.Tensor:
    """Spearman correlation between the rows of two rank matrices, ``(M, C)``."""
    a = rank_a - rank_a.mean(dim=1, keepdim=True)
    b = rank_b - rank_b.mean(dim=1, keepdim=True)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return a @ b.t()

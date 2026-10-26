"""Cross-domain rank signatures (``L_rm``).

Every labelled source class accumulates the mean *rank vector* of its pixels over a set of
*class anchors* (the ranks of a pixel's similarities to the class anchors).  A target pixel is
matched to the class signature its own rank vector correlates with most strongly (Spearman's rho),
and the matched class supervises the student.

The class anchors are identity-preserving momentum anchors: the anchors of class ``c`` are initialised by
the seeded orthogonal selection inside that class and are then tracked by an online spherical
k-means in the space of the EMA teacher.  At every step each typical source pixel of class ``c``
(cosine to the class mean direction in the top ``1 - zeta`` fraction) is assigned to the most
similar anchor of class ``c``, and every anchor moves towards the normalised mean of its pixels with
momentum 0.9999; a class absent from the batch keeps its anchors.  One pixel feeds one anchor, so
anchor ``k`` keeps covering the same structure and the class signatures (momentum 0.9) stay defined
in a coherent coordinate system while the features move.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .anchors import rank_correlation, row_rank, seeded_ortho

__all__ = ["SignatureBank", "RankMatcher", "downsample_labels"]


def downsample_labels(lab: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(lab.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1).long()


def _rank_in_class(sorted_cls: torch.Tensor, n_cls: int) -> torch.Tensor:
    """position of every element inside its class block of a class-sorted vector."""
    cnt = torch.zeros(n_cls, device=sorted_cls.device, dtype=torch.long).index_add_(0, sorted_cls, torch.ones_like(sorted_cls))
    start = torch.cumsum(cnt, 0) - cnt
    return torch.arange(sorted_cls.shape[0], device=sorted_cls.device) - start[sorted_cls]


class SignatureBank:
    """Class anchors (momentum anchors) plus an exponential moving average of the per-class rank signatures."""

    def __init__(self, num_classes: int, zeta: float = 0.7, max_per_class: int = 20000, max_track: int = 4096):
        self.num_classes = int(num_classes)
        self.zeta = float(zeta)
        self.max_per_class = int(max_per_class)
        self.max_track = int(max_track)
        self.anchors: torch.Tensor | None = None
        self.anchor_cls: torch.Tensor | None = None
        self.sig: torch.Tensor | None = None
        self.seen: torch.Tensor | None = None

    def ready(self) -> bool:
        return self.sig is not None and self.seen is not None and bool(self.seen.min() > 0)

    @torch.no_grad()
    def init_anchors(self, feats: torch.Tensor, labels: torch.Tensor, k: int) -> None:
        """``k / num_classes`` seeded orthogonal anchors inside every source class (remainder to the largest classes)."""
        present = [c for c in range(self.num_classes) if bool((labels == c).any())]
        per = {c: k // len(present) for c in present}
        for c in sorted(present, key=lambda c: -int((labels == c).sum()))[: k - sum(per.values())]:
            per[c] += 1
        anchors, cls = [], []
        for c in present:
            xc = feats[labels == c]
            if xc.shape[0] > self.max_per_class:
                g = torch.Generator(device="cpu").manual_seed(c)
                xc = xc[torch.randperm(xc.shape[0], generator=g)[: self.max_per_class].to(xc.device)]
            a = seeded_ortho(xc, per[c], self.zeta)
            anchors.append(a)
            cls.append(torch.full((a.shape[0],), c, device=a.device, dtype=torch.long))
        self.anchors = F.normalize(torch.cat(anchors), dim=1)
        self.anchor_cls = torch.cat(cls)

    @torch.no_grad()
    def track(self, feats: torch.Tensor, labels: torch.Tensor, generator: torch.Generator | None = None) -> None:
        """One momentum step of the class-wise online spherical k-means; ``feats`` are unit rows."""
        n_cls, dev = self.num_classes, feats.device
        has = torch.zeros(n_cls + 2, dtype=torch.bool, device=dev)
        has[self.anchor_cls] = True
        y = torch.where((labels >= 0) & (labels < n_cls), labels, torch.full_like(labels, n_cls))
        y = torch.where(has[y], y, torch.full_like(y, n_cls))           # class n_cls: pixels without anchors
        # at most max_track random pixels per class
        prio = torch.rand(y.shape[0], generator=generator).to(dev)
        order = torch.argsort(y.to(torch.float64) * 2.0 + prio.to(torch.float64))
        rank = _rank_in_class(y[order], n_cls + 2)
        sel = torch.zeros_like(y, dtype=torch.bool).scatter_(0, order, rank < self.max_track) & (y < n_cls)
        # typical pixels: cosine to the class mean direction in the top (1 - zeta) fraction of the selected pixels
        w = sel.to(feats.dtype)
        mu = F.normalize(torch.zeros(n_cls + 2, feats.shape[1], device=dev, dtype=feats.dtype)
                         .index_add_(0, y, feats * w[:, None]), dim=1)
        typ = (feats * mu[y]).sum(1)
        y2 = torch.where(sel, y, torch.full_like(y, n_cls + 1))
        order = torch.argsort(y2.to(torch.float64) * 4.0 + typ.to(torch.float64) + 1.0)
        ys = y2[order]
        cnt = torch.zeros(n_cls + 2, device=dev, dtype=torch.float64).index_add_(0, ys, torch.ones_like(ys, dtype=torch.float64))
        rank = _rank_in_class(ys, n_cls + 2)
        keep_sorted = (ys < n_cls) & (rank >= torch.ceil(self.zeta * (cnt[ys] - 1.0)).long())
        keep = torch.zeros_like(sel).scatter_(0, order, keep_sorted)
        # nearest anchor of the pixel's own class; every anchor moves towards the mean of its pixels
        sim = (feats @ self.anchors.t()).masked_fill(self.anchor_cls[None, :] != y[:, None], float("-inf"))
        assign = sim.argmax(dim=1)
        w = keep.to(feats.dtype)
        k = self.anchors.shape[0]
        sums = torch.zeros(k, feats.shape[1], device=dev, dtype=feats.dtype).index_add_(0, assign, feats * w[:, None])
        hit = torch.zeros(k, device=dev, dtype=feats.dtype).index_add_(0, assign, w) > 0
        moved = F.normalize(0.9999 * self.anchors + (1.0 - 0.9999) * F.normalize(sums, dim=1), dim=1)
        self.anchors = torch.where(hit[:, None], moved, self.anchors)

    @torch.no_grad()
    def similarities(self, fea: torch.Tensor) -> torch.Tensor:
        b, c, h, w = fea.shape
        f = F.normalize(fea.reshape(b, c, -1), p=2, dim=1)
        rel = torch.einsum("bcn,kc->bnk", f, self.anchors)
        return rel.reshape(b, h, w, self.anchors.shape[0])

    @torch.no_grad()
    def update(self, fea_src: torch.Tensor, gt_src: torch.Tensor) -> None:
        rel = self.similarities(fea_src)
        vec = row_rank(rel.reshape(-1, rel.shape[-1]))
        n = self.num_classes
        g = gt_src.reshape(-1)
        g = torch.where((g >= 0) & (g < n), g, torch.full_like(g, n))
        if self.sig is None:
            self.sig = torch.zeros(n, vec.shape[1], device=vec.device, dtype=vec.dtype)
            self.seen = torch.zeros(n, device=vec.device, dtype=vec.dtype)
        cnt = torch.zeros(n + 1, device=vec.device, dtype=vec.dtype).index_add_(0, g, torch.ones_like(g, dtype=vec.dtype))[:n]
        cur = torch.zeros(n + 1, vec.shape[1], device=vec.device, dtype=vec.dtype).index_add_(0, g, vec)[:n]
        cur = cur / cnt.clamp_min(1.0)[:, None]
        present = (cnt > 0)[:, None]
        ema = torch.where(self.seen[:, None] > 0, 0.9 * self.sig + (1.0 - 0.9) * cur, cur)
        self.sig = torch.where(present, ema, self.sig)
        self.seen = torch.where(present[:, 0], torch.ones_like(self.seen), self.seen)

    @torch.no_grad()
    def match(self, fea_tgt: torch.Tensor):
        """Matched class, margin weight and highest correlation of every target pixel."""
        b, c, h, w = fea_tgt.shape
        rel = self.similarities(fea_tgt)
        scores = rank_correlation(row_rank(rel.reshape(-1, rel.shape[-1])), row_rank(self.sig))
        top2 = scores.topk(2, dim=1)
        label = top2.indices[:, 0]
        margin = (0.5 * (top2.values[:, 0] - top2.values[:, 1])).clamp(0.0, 1.0)
        rho = scores.max(dim=1).values
        return label.reshape(b, h, w).long(), margin.reshape(b, h, w), rho.reshape(b, h, w)

    def state_dict(self) -> dict:
        return {k: (None if getattr(self, k) is None else getattr(self, k).detach().cpu())
                for k in ("anchors", "anchor_cls", "sig", "seen")}

    def load_state_dict(self, state: dict | None, device: torch.device) -> None:
        if not state:
            return
        for k in ("anchors", "anchor_cls", "sig", "seen"):
            setattr(self, k, None if state.get(k) is None else state[k].to(device))


class RankMatcher:
    """Owns the momentum-anchor signature bank and the ``L_rm`` objective.

    Source and target features are read with the EMA teacher; the anchors are initialised from the
    first 8 source batches and tracked at every following step.
    """

    def __init__(self, num_classes: int, num_anchors: int, quantile: float, seed: int = 0, init_pixels: int = 60000,
                 track_pixels: int = 20000):
        self.bank = SignatureBank(num_classes, zeta=quantile)
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)
        self.init_pixels = int(init_pixels)
        self.track_pixels = int(track_pixels)
        self.generator = torch.Generator().manual_seed(int(seed) + 17)
        self._buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._last_iter = None

    @torch.no_grad()
    def update(self, it: int, src_feat: torch.Tensor, src_lab: torch.Tensor) -> None:
        """Initialise or track the momentum anchors, then accumulate the source signatures."""
        if self._last_iter == it:
            return
        self._last_iter = it
        feat = src_feat.detach()
        lab = downsample_labels(src_lab, feat.shape[-2:])
        flat_f = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        n_keep = self.init_pixels if self.bank.anchors is None else self.track_pixels
        sel = torch.randperm(flat_f.shape[0], generator=self.generator)[:n_keep].to(flat_f.device)
        f, fl = F.normalize(flat_f[sel], dim=1), lab.reshape(-1)[sel]
        if self.bank.anchors is None:
            self._buffer.append((f, fl))
            if len(self._buffer) < 8:
                return
            self.bank.init_anchors(torch.cat([b[0] for b in self._buffer]), torch.cat([b[1] for b in self._buffer]),
                                   self.num_anchors)
            self._buffer = []
        else:
            self.bank.track(f, fl, self.generator)
        self.bank.update(feat, lab)

    def loss(self, tgt_feat: torch.Tensor, student_logits: torch.Tensor):
        """Margin-weighted cross-entropy towards the matched source class.

        In every matched class only its 30% best-correlated pixels are kept.
        """
        zero = student_logits.sum() * 0.0
        if not self.bank.ready():
            return zero, 0.0
        label, margin, rho = self.bank.match(tgt_feat.detach())
        n = self.num_classes
        lab_f, rho_f = label.reshape(-1), rho.reshape(-1)
        ones = torch.ones_like(lab_f, dtype=torch.float64)
        m_cnt = torch.zeros(n, device=lab_f.device, dtype=torch.float64).index_add_(0, lab_f, ones)
        budget = torch.floor((1.0 - 0.7) * m_cnt).long()
        # threshold of every class = its budget-th largest correlation; every pixel reaching it is kept
        order = torch.argsort(lab_f.to(torch.float64) * 4.0 + (1.0 - rho_f.to(torch.float64)))
        start = (torch.cumsum(m_cnt, 0) - m_cnt).long()
        thr = rho_f[order][(start + budget - 1).clamp(0, lab_f.numel() - 1)]
        keep = ((budget[lab_f] > 0) & (rho_f >= thr[lab_f])).reshape(label.shape)
        margin = torch.where(keep, margin, torch.zeros_like(margin))
        label_up = downsample_labels(label, student_logits.shape[-2:])
        w_up = F.interpolate(margin.unsqueeze(1), size=student_logits.shape[-2:], mode="bilinear",
                             align_corners=False).squeeze(1).detach()
        ce = F.cross_entropy(student_logits, label_up.clamp_max(student_logits.shape[1] - 1), reduction="none")
        return (ce * w_up).mean(), float((w_up > 0).float().mean())

    def state_dict(self) -> dict:
        state = self.bank.state_dict()
        state["generator"] = self.generator.get_state()
        state["buffer"] = [(f.detach().cpu(), l.detach().cpu()) for f, l in self._buffer]
        return state

    def load_state_dict(self, state: dict | None, device: torch.device) -> None:
        if not state:
            return
        self.bank.load_state_dict(state, device)
        if state.get("generator") is not None:
            self.generator.set_state(state["generator"])
        self._buffer = [(f.to(device), l.to(device)) for f, l in state.get("buffer", [])]

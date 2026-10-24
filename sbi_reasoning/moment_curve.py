"""Moment-curve mass estimator from three templates.

A linear (multinomial logistic) classifier is trained on the three templates. Its mean
log-probability differences

    u = (<log p_1 - log p_0>, <log p_2 - log p_1>)

are exactly means of linear functions of the input features, so E_m[u] is a smooth curve in the
parent mass m. The curve is measured at the template masses on held-out events (k-fold
cross-fitting, so every template event is used once), each component is interpolated by the
quadratic through the three anchors, and a dataset's u is projected onto that curve:

    m_hat = argmin_m |u - q(m)|^2   over a window around the templates.

Unlike a quadratic fitted to the template log-likelihoods, this assumes no shape for the
likelihood, only that E_m[u] is quadratic in m between the templates.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from .models import DenseNetwork, Standardizer
from .physics import reconstruct_parent_masses


def visible_features(observables: np.ndarray) -> np.ndarray:
    """The 10 observables plus 10 rotation-invariant summaries of them."""
    l1, l2, met = observables[:, 0:4], observables[:, 4:8], observables[:, 8:10]

    def pt(v):
        return np.hypot(v[:, 1], v[:, 2])

    def pmag(v):
        return np.linalg.norm(v[:, 1:4], axis=1)

    ll = l1 + l2
    mll = np.sqrt(np.maximum(ll[:, 0] ** 2 - np.sum(ll[:, 1:] ** 2, axis=1), 0.0))
    cos12 = np.sum(l1[:, 1:] * l2[:, 1:], axis=1) / (pmag(l1) * pmag(l2) + 1e-9)
    summaries = np.stack(
        [
            np.log(l1[:, 0]),
            np.log(l2[:, 0]),
            np.log(pt(l1)),
            np.log(pt(l2)),
            np.log(mll + 1e-3),
            cos12,
            np.log(np.hypot(met[:, 0], met[:, 1]) + 1e-3),
            np.log(l1[:, 0] + l2[:, 0]),
            np.log(np.minimum(l1[:, 0], l2[:, 0])),
            np.log(np.maximum(l1[:, 0], l2[:, 0])),
        ],
        axis=1,
    )
    return np.concatenate([observables, summaries], axis=1).astype(np.float32)


def full_features(observables: np.ndarray, neutrinos: np.ndarray) -> np.ndarray:
    """Visible features plus the neutrino-dependent invariants: both lepton+neutrino parent
    masses, their min and max, and the neutrino energies and transverse momenta."""
    neutrinos = neutrinos.reshape(-1, 2, 3)
    masses = reconstruct_parent_masses(observables[:, :8].reshape(-1, 2, 4), neutrinos, 0.0)
    energy = np.linalg.norm(neutrinos, axis=-1)
    pt = np.hypot(neutrinos[..., 0], neutrinos[..., 1])
    extra = np.log1p(
        np.concatenate(
            [masses, masses.min(1, keepdims=True), masses.max(1, keepdims=True), energy, pt], axis=1
        )
    )
    return np.concatenate([visible_features(observables), extra], axis=1).astype(np.float32)


def _fit_linear(x, y, epochs, device, seed):
    """Multinomial logistic regression with the profiler's optimiser settings."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = DenseNetwork(x.shape[1], 3, []).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    for _ in range(epochs):
        order = torch.randperm(len(xt), generator=generator).to(device)
        for start in range(0, len(xt), 2048):
            index = order[start : start + 2048]
            loss = F.cross_entropy(model(xt[index]), yt[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    layer = model.network[0]
    return (
        layer.weight.detach().cpu().numpy().astype(np.float64),
        layer.bias.detach().cpu().numpy().astype(np.float64),
    )


class MomentCurve:
    """`featurize` maps the estimator's inputs (observables, or observables and neutrinos) to
    features; `margin` is how far outside the template range the projection may go."""

    def __init__(
        self,
        featurize: Callable[..., np.ndarray],
        template_masses: list[float],
        margin_gev: float = 50.0,
    ):
        self.featurize = featurize
        self.masses = np.asarray(template_masses, dtype=np.float64)
        self.window = np.arange(self.masses.min() - margin_gev, self.masses.max() + margin_gev + 1e-9, 0.1)
        self.folds: list[tuple[Standardizer, np.ndarray, np.ndarray]] = []
        self.anchors: np.ndarray | None = None
        self.curve: np.ndarray | None = None

    @staticmethod
    def _diffs(phi, fold):
        scaler, weight, bias = fold
        logits = scaler.transform(phi).astype(np.float64) @ weight.T + bias
        return np.stack([logits[:, 1] - logits[:, 0], logits[:, 2] - logits[:, 1]], axis=1)

    def fit(self, inputs, labels, folds=5, epochs=100, device=torch.device("cpu"), seed=0):
        """`inputs`: tuple of arrays passed to `featurize`; `labels`: template index 0/1/2."""
        phi = self.featurize(*inputs)
        fold_id = np.empty(len(labels), dtype=np.int64)
        rng = np.random.default_rng([seed, 17])
        for c in range(3):
            index = rng.permutation(np.flatnonzero(labels == c))
            fold_id[index] = np.arange(len(index)) % folds
        held_out = np.zeros((len(labels), 2))
        for k in range(folds):
            train = fold_id != k
            scaler = Standardizer.fit(phi[train])
            weight, bias = _fit_linear(scaler.transform(phi[train]), labels[train], epochs, device, seed * 100 + k)
            self.folds.append((scaler, weight, bias))
            held_out[~train] = self._diffs(phi[~train], self.folds[-1])
        self.anchors = np.array([held_out[labels == c].mean(0) for c in range(3)])
        if not np.all(np.isfinite(self.anchors)):
            raise RuntimeError(f"non-finite moment-curve anchors: {self.anchors}")
        self.curve = np.stack(
            [np.polyval(np.polyfit(self.masses, self.anchors[:, j], 2), self.window) for j in range(2)], axis=1
        )
        return self

    def statistic(self, *inputs) -> np.ndarray:
        phi = self.featurize(*inputs)
        return np.mean([self._diffs(phi, fold).mean(0) for fold in self.folds], axis=0)

    def estimate(self, *inputs) -> float:
        u = self.statistic(*inputs)
        return float(self.window[np.argmin(np.sum((u[None, :] - self.curve) ** 2, axis=1))])

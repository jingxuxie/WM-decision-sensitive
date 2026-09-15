"""Exact decision-sensitive compression for whitened quadratic planning.

Model: y = D x + B u, E[x x.T] = I; cost ||y||^2 + ||u||^2.
The rank budget applies to the state/history-to-future map, not to B.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

Array = NDArray[np.float64]


def _matrix(a: Array, name: str) -> Array:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite matrix")
    return a


def controller(b: Array) -> tuple[Array, Array]:
    b = _matrix(b, "B")
    h = np.eye(b.shape[1]) + b.T @ b
    return np.linalg.solve(h, b.T), h


def metric(b: Array) -> Array:
    k, _ = controller(b)
    w = b @ k
    return (w + w.T) / 2


def psd_sqrt(w: Array) -> Array:
    values, vectors = eigh((w + w.T) / 2)
    if values.min() < -1e-8:
        raise ValueError("matrix is not positive semidefinite")
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T


def spectral_compress(d: Array, w: Array, rank: int) -> tuple[Array, Array]:
    d = _matrix(d, "D")
    w = _matrix(w, "W")
    if w.shape != (d.shape[0], d.shape[0]):
        raise ValueError("metric/output dimension mismatch")
    if not 0 <= rank <= d.shape[1]:
        raise ValueError("rank outside input dimension")
    if rank == 0:
        p = np.zeros((d.shape[1], d.shape[1]))
    elif rank == d.shape[1]:
        p = np.eye(d.shape[1])
    else:
        _, v = eigh((d.T @ w @ d + d.T @ w.T @ d) / 2)
        v = v[:, -rank:]
        p = v @ v.T
    return d @ p, p


def regret(d: Array, b: Array, f: Array, b_hat: Array) -> float:
    """Exact expected excess cost; no subtraction of nearly equal costs."""
    k, h = controller(b)
    kh, _ = controller(b_hat)
    error = kh @ f - k @ d
    return float(np.sum(error * (h @ error)))


def compression_loss(d: Array, b: Array, p: Array) -> float:
    residual = d @ (np.eye(d.shape[1]) - p)
    return float(np.sum(residual * (metric(b) @ residual)))


@dataclass(frozen=True)
class Radii:
    forecast_fro: float
    action_op: float
    delta: float

    @property
    def eta(self) -> float:
        return min(1.0, self.action_op)


def gaussian_radii(p: int, d: int, m: int, gram: Array,
                   sigma: float, delta: float = 0.05) -> Radii:
    """Fixed-design Gaussian OLS bounds; all ranks share the same event.

    gram = Z Z.T, with state/history features before action features.
    Known, independent N(0,sigma^2) output noise is required.
    """
    if not (0 < delta < 1) or sigma < 0:
        raise ValueError("invalid noise/confidence parameters")
    if gram.shape != (d + m, d + m):
        raise ValueError("incorrect design Gram shape")
    if np.linalg.eigvalsh(gram).min() <= 0:
        raise ValueError("full-rank design required; no missing-coverage certificate")
    inv = np.linalg.inv(gram)
    cd = np.linalg.eigvalsh(inv[:d, :d]).max()
    cb = np.linalg.eigvalsh(inv[d:, d:]).max()
    tail = np.sqrt(2 * np.log(2 / delta))
    ed = sigma * np.sqrt(cd) * (np.sqrt(p * d) + tail)
    eb = sigma * np.sqrt(cb) * (np.sqrt(p) + np.sqrt(m) + tail)
    return Radii(float(ed), float(eb), delta)


def certificate(d_hat: Array, b_hat: Array, f: Array,
                radii: Radii) -> float:
    residual = f - d_hat
    upper = metric(b_hat) + radii.eta * np.eye(d_hat.shape[0])
    q = max(0.0, float(np.sum(residual * (upper @ residual))))
    root = np.sqrt(q) + radii.forecast_fro
    root += 1.5 * radii.action_op * np.linalg.norm(f, "fro")
    return float(root ** 2)


def oracle_bound(d: Array, b: Array, d_hat: Array,
                 rank: int, radii: Radii) -> float:
    _, p = spectral_compress(d, metric(b), rank)
    tail = d @ (np.eye(d.shape[1]) - p)
    loss = compression_loss(d, b, p)
    root = np.sqrt(loss + 2 * radii.eta * np.linalg.norm(tail, "fro") ** 2)
    root += (1 + np.sqrt(1 + radii.eta)) * radii.forecast_fro
    root += 1.5 * radii.action_op * np.linalg.norm(d_hat, "fro")
    return float(root ** 2)


def fit_ols(z: Array, y: Array) -> Array:
    """Columns are independently reset regression observations."""
    if z.ndim != 2 or y.ndim != 2 or z.shape[1] != y.shape[1]:
        raise ValueError("incompatible regression matrices")
    gram = z @ z.T
    if np.linalg.matrix_rank(gram) != gram.shape[0]:
        raise ValueError("OLS design lacks coverage")
    return np.linalg.solve(gram, z @ y.T).T


def draw_ols(d: Array, b: Array, n: int, sigma: float, coverage: float,
             rng: np.random.Generator) -> tuple[Array, Array, Array]:
    """Draw exact sufficient statistics, not an approximation to training.

    Fixed reset-probe design: ZZ.T = n diag(I_d, coverage I_m).
    It exists for n >= d+m. Gaussian OLS errors then have this exact law.
    """
    p, dim = d.shape
    m = b.shape[1]
    if n < dim + m or coverage <= 0:
        raise ValueError("insufficient sample count or action coverage")
    scales = np.r_[np.ones(dim), np.full(m, coverage)]
    estimate = np.c_[d, b] + rng.normal(size=(p, dim + m)) * sigma / np.sqrt(n * scales)
    return estimate[:, :dim], estimate[:, dim:], np.diag(n * scales)


def lift_lti(a: Array, b: Array, horizon: int,
             q_sqrt: Array | None = None, action_cost: float = 1.0) -> tuple[Array, Array]:
    """Stack x_1,...,x_H and whiten a scalar positive action penalty."""
    d, m = b.shape
    if horizon < 1 or action_cost <= 0:
        raise ValueError("invalid horizon or action penalty")
    c = np.eye(d) if q_sqrt is None else q_sqrt
    powers = [np.eye(d)]
    for _ in range(horizon):
        powers.append(powers[-1] @ a)
    ds = np.vstack([c @ powers[k] for k in range(1, horizon + 1)])
    bs = np.zeros((horizon * c.shape[0], horizon * m))
    for i in range(horizon):
        for j in range(i + 1):
            bs[i*c.shape[0]:(i+1)*c.shape[0], j*m:(j+1)*m] = c @ powers[i-j] @ b / np.sqrt(action_cost)
    return ds, bs

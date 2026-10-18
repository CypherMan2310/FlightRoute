"""Two-layer graph convolutional network in numpy/scipy (no torch). Train offline, run anywhere.

Model (per snapshot, X in R^{N x F}, Ahat in R^{N x N} sparse):
    Z = (Ahat X) W1 + b1,   H = relu(Z),   Yhat = Ahat (H W2) + b2        (2 heads: departure delay, late-aircraft delay)
Loss: sum_{b,n,k} mw * huber_delta(Yhat - Y) / sum(mw)  +  lam (|W1|^2 + |W2|^2), with mw = mask * weight.
There are no node-id features, so trained weights run on any graph (the world graph at inference).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse


def operator_inflow(flows: sparse.spmatrix) -> sparse.csr_matrix:
    """Ahat = 1/2 I + 1/2 D_in^-1 A^T where flows[j, i] = traffic j -> i: half self, half flight-weighted mean of upstream
    airports. Row-stochastic. A node with no inbound traffic keeps only itself."""
    a = sparse.csr_matrix(flows, dtype=np.float64)
    n = a.shape[0]
    d_in = np.asarray(a.sum(axis=0)).ravel()
    inv = np.divide(1.0, d_in, out=np.zeros_like(d_in), where=d_in > 0)
    mean_in = sparse.diags(inv) @ a.T
    self_w = np.where(d_in > 0, 0.5, 1.0)
    return (sparse.diags(self_w) + 0.5 * mean_in).tocsr()


def operator_symmetric(adj: sparse.spmatrix) -> sparse.csr_matrix:
    """Kipf & Welling: D^-1/2 (A + I) D^-1/2."""
    a = sparse.csr_matrix(adj, dtype=np.float64) + sparse.identity(adj.shape[0], format="csr")
    d = 1.0 / np.sqrt(np.asarray(a.sum(axis=1)).ravel())
    return (sparse.diags(d) @ a @ sparse.diags(d)).tocsr()


def save_operator(path, ahat: sparse.spmatrix) -> None:
    sparse.save_npz(path, sparse.csr_matrix(ahat))


def load_operator(path) -> sparse.csr_matrix:
    return sparse.load_npz(path).tocsr()


class Standardizer:
    """z = (x - mean) / std. A feature that never varied in the fit data gets std = inf, so it is always 0: a level the model never
    saw (a zone one-hot for a climate absent from training) cannot move a prediction."""

    def fit(self, x: np.ndarray) -> "Standardizer":
        flat = x.reshape(-1, x.shape[-1])
        self.mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        self.std = np.where(std < 1e-8, np.inf, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def _smul(a: sparse.spmatrix, t: np.ndarray) -> np.ndarray:
    """Apply a sparse (N x N) operator along the node axis of t (B, N, K)."""
    b, n, k = t.shape
    out = a @ t.transpose(1, 0, 2).reshape(n, b * k)
    return np.asarray(out).reshape(n, b, k).transpose(1, 0, 2)


def _huber(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


class GCN:
    def __init__(self, n_features: int, hidden: int = 32, n_out: int = 2, seed: int = 0, dtype=np.float64):
        rng = np.random.default_rng(seed)
        self.n_features, self.hidden, self.n_out = n_features, hidden, n_out
        self.W1 = (rng.normal(size=(n_features, hidden)) * np.sqrt(2.0 / n_features)).astype(dtype)  # He
        self.b1 = np.zeros(hidden, dtype=dtype)
        self.W2 = (rng.normal(size=(hidden, n_out)) * np.sqrt(1.0 / hidden)).astype(dtype)  # Xavier-ish
        self.b2 = np.zeros(n_out, dtype=dtype)
        self.scaler: Standardizer | None = None

    PARAMS = ("W1", "b1", "W2", "b2")

    def forward(self, ahat, x: np.ndarray):
        """x is (B, N, F) and already standardised. Returns (Yhat, cache)."""
        p = _smul(ahat, x)
        z = p @ self.W1 + self.b1
        h = np.maximum(z, 0.0)
        ah = _smul(ahat, h)
        return ah @ self.W2 + self.b2, {"P": p, "Z": z, "H": h, "AH": ah}

    def loss_and_grads(self, ahat, x, y, mw, lam: float = 1e-4, delta: float = 15.0):
        yhat, c = self.forward(ahat, x)
        r = yhat - y
        s = max(float(mw.sum()), 1e-12)
        loss = float((mw[..., None] * _huber(r, delta)).sum() / s + lam * ((self.W1**2).sum() + (self.W2**2).sum()))
        g = mw[..., None] * np.clip(r, -delta, delta) / s
        d_h = _smul(ahat.T, g) @ self.W2.T
        d_z = d_h * (c["Z"] > 0)
        grads = {
            "W2": np.einsum("bnh,bno->ho", c["AH"], g) + 2 * lam * self.W2,
            "b2": g.sum(axis=(0, 1)),
            "W1": np.einsum("bnf,bnh->fh", c["P"], d_z) + 2 * lam * self.W1,
            "b1": d_z.sum(axis=(0, 1)),
        }
        return loss, grads

    def _val_mae(self, ahat, val) -> float:
        xv, yv, mv, wv = val
        pred = self.forward(ahat, self._prep(xv))[0]
        mw = mv * wv
        return float((mw[..., None] * np.abs(pred - yv)).sum() / max(float(mw.sum()) * pred.shape[-1], 1e-12))

    def _prep(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x) if self.scaler is not None else x

    def fit(self, ahat, x, y, m, w, *, val=None, scaler: Standardizer | None = None, epochs=100, batch=32, lr=1e-3, patience=10, delta=15.0, lam=1e-4, seed=0) -> dict:
        """Adam on mini-batches of snapshots. x raw (B, N, F). Early stopping on `val` MAE (restores the best weights)."""
        rng = np.random.default_rng(seed)
        self.scaler = scaler if scaler is not None else Standardizer().fit(x)
        xs, mw = self._prep(x), m * w
        self.b2 = ((mw[..., None] * y).sum(axis=(0, 1)) / max(float(mw.sum()), 1e-12)).astype(self.b2.dtype)
        mom = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        var = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        hist: dict[str, list] = {"train_loss": [], "val_mae": []}
        best, best_params, bad, step = np.inf, None, 0, 0
        for _ in range(epochs):
            order = rng.permutation(len(xs))
            losses = []
            for i in range(0, len(order), batch):
                idx = order[i : i + batch]
                loss, grads = self.loss_and_grads(ahat, xs[idx], y[idx], mw[idx], lam=lam, delta=delta)
                step += 1
                for k in self.PARAMS:
                    mom[k] = 0.9 * mom[k] + 0.1 * grads[k]
                    var[k] = 0.999 * var[k] + 0.001 * grads[k] ** 2
                    mh, vh = mom[k] / (1 - 0.9**step), var[k] / (1 - 0.999**step)
                    getattr(self, k)[...] -= lr * mh / (np.sqrt(vh) + 1e-8)
                losses.append(loss)
            hist["train_loss"].append(float(np.mean(losses)))
            if val is not None:
                v = self._val_mae(ahat, val)
                hist["val_mae"].append(v)
                if v < best - 1e-9:
                    best, bad, best_params = v, 0, {k: getattr(self, k).copy() for k in self.PARAMS}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_params is not None:
            for k, v in best_params.items():
                getattr(self, k)[...] = v
        return hist

    def predict(self, ahat, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 2
        out = self.forward(ahat, self._prep(x[None] if single else x))[0]
        return out[0] if single else out

    def save(self, path, *, feature_names, nodes=None) -> None:
        assert self.scaler is not None, "fit (or set a scaler) before saving"
        np.savez_compressed(
            path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, mean=self.scaler.mean, std=self.scaler.std,
            feature_names=np.array(list(feature_names), dtype="U64"), nodes=np.array(list(nodes or []), dtype="U16"),
        )

    @classmethod
    def load(cls, path) -> tuple["GCN", dict]:
        z = np.load(Path(path), allow_pickle=False)
        model = cls(z["W1"].shape[0], hidden=z["W1"].shape[1], n_out=z["W2"].shape[1])
        for k in cls.PARAMS:
            getattr(model, k)[...] = z[k]
        model.scaler = Standardizer()
        model.scaler.mean, model.scaler.std = z["mean"], z["std"]
        return model, {"feature_names": [str(s) for s in z["feature_names"]], "nodes": [str(s) for s in z["nodes"]]}

import numpy as np
import pytest
from scipy import sparse

from aerogrid.ml import gcn


def ring_flows(n):
    """Directed ring j -> j+1 with weight j+1, plus a chord, so the in-flow operator is not trivial."""
    a = np.zeros((n, n))
    for j in range(n):
        a[j, (j + 1) % n] = j + 1
    a[0, n // 2] = 3
    return sparse.csr_matrix(a)


def test_inflow_operator_is_row_stochastic_with_half_self_weight():
    ahat = gcn.operator_inflow(ring_flows(6)).toarray()
    assert np.allclose(ahat.sum(axis=1), 1.0)
    assert np.allclose(np.diag(ahat), 0.5)
    # node 1 is fed by node 0 only (weight 1): half of its row goes to node 0
    assert ahat[1, 0] == pytest.approx(0.5)
    # node 3 is fed by node 0 (chord, weight 3) and node 2 (weight 3): split evenly
    assert ahat[3, 0] == pytest.approx(0.25) and ahat[3, 2] == pytest.approx(0.25)


def test_inflow_operator_isolated_node_keeps_only_itself():
    a = sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0.0]]))
    ahat = gcn.operator_inflow(a).toarray()
    assert np.allclose(ahat[0], [1, 0, 0]) and np.allclose(ahat[2], [0, 0, 1])
    assert np.allclose(ahat[1], [0.5, 0.5, 0])


def test_symmetric_operator_matches_kipf():
    a = sparse.csr_matrix(np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0.0]]))
    s = gcn.operator_symmetric(a).toarray()
    assert np.allclose(s, s.T)
    deg = np.array([3.0, 4.0, 2.0])  # row sums of A + I
    assert s[0, 1] == pytest.approx(2 / np.sqrt(deg[0] * deg[1]))
    assert s[0, 0] == pytest.approx(1 / deg[0])


def _problem(seed, delta):
    rng = np.random.default_rng(seed)
    n, f, h, b = 6, 3, 4, 3
    ahat = gcn.operator_inflow(sparse.csr_matrix(rng.random((n, n)) * (rng.random((n, n)) < 0.6)))
    x = rng.normal(size=(b, n, f))
    y = rng.normal(scale=2.0, size=(b, n, 2))
    m = (rng.random((b, n)) < 0.85).astype(float)
    w = rng.uniform(0.2, 1.0, size=(b, n))
    model = gcn.GCN(f, hidden=h, seed=seed, dtype=np.float64)
    return ahat, x, y, m, w, model


def _kink_free(model, ahat, x, y, m, w, delta):
    yhat, cache = model.forward(ahat, x)
    z_ok = np.abs(cache["Z"]).min() > 1e-4
    r_ok = np.abs(np.abs(yhat - y) - delta).min() > 1e-4
    return z_ok and r_ok


def test_gradients_match_central_differences():
    delta, lam, eps = 1.0, 1e-3, 1e-6
    for seed in range(50):
        ahat, x, y, m, w, model = _problem(seed, delta)
        if _kink_free(model, ahat, x, y, m, w, delta):
            break
    else:
        pytest.fail("no kink-free draw found")
    loss, grads = model.loss_and_grads(ahat, x, y, m * w, lam=lam, delta=delta)
    assert np.isfinite(loss)
    for name in ("W1", "b1", "W2", "b2"):
        p = getattr(model, name)
        num = np.zeros_like(p)
        for idx in np.ndindex(*p.shape):
            old = p[idx]
            p[idx] = old + eps
            lp, _ = model.loss_and_grads(ahat, x, y, m * w, lam=lam, delta=delta)
            p[idx] = old - eps
            lm, _ = model.loss_and_grads(ahat, x, y, m * w, lam=lam, delta=delta)
            p[idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        rel = np.abs(num - grads[name]) / np.maximum(1e-12, np.abs(num) + np.abs(grads[name]))
        assert rel.max() < 1e-6, (name, rel.max())


def _toy(seed=0, b=96, n=12, f=4):
    """Targets come from a random 'teacher' GCN of the same form, so a student with the graph can fit them and one without cannot."""
    rng = np.random.default_rng(seed)
    ahat = gcn.operator_inflow(ring_flows(n))
    x = rng.normal(size=(b, n, f))
    teacher = gcn.GCN(f, hidden=8, seed=7)
    teacher.W2 *= 10
    y = teacher.forward(ahat, x)[0] + rng.normal(scale=0.3, size=(b, n, 2))
    ones = np.ones((b, n))
    return ahat, x, y, ones


def _mae(pred, y):
    return float(np.abs(pred - y).mean())


def test_training_learns_neighbour_dependent_targets_and_graph_matters():
    ahat, x, y, ones = _toy()
    tr, va = slice(0, 72), slice(72, 96)
    scaler = gcn.Standardizer().fit(x[tr])
    baseline = _mae(np.broadcast_to(y[tr].mean(axis=(0, 1)), y[va].shape), y[va])

    model = gcn.GCN(4, hidden=16, seed=1)
    hist = model.fit(ahat, x[tr], y[tr], ones[tr], ones[tr], val=(x[va], y[va], ones[va], ones[va]), scaler=scaler, epochs=200, batch=24, lr=1e-2, patience=30, delta=15.0, seed=1)
    assert hist["val_mae"][-1] < hist["val_mae"][0]
    with_graph = _mae(model.predict(ahat, x[va]), y[va])
    assert with_graph < 0.35 * baseline

    no_graph = gcn.GCN(4, hidden=16, seed=1)
    ident = sparse.identity(12, format="csr")
    no_graph.fit(ident, x[tr], y[tr], ones[tr], ones[tr], val=(x[va], y[va], ones[va], ones[va]), scaler=scaler, epochs=200, batch=24, lr=1e-2, patience=30, delta=15.0, seed=1)
    assert with_graph < _mae(no_graph.predict(ident, x[va]), y[va])


def test_masked_nodes_do_not_influence_training():
    ahat, x, y, ones = _toy(b=24)
    mask = ones.copy()
    mask[:, 5] = 0
    y_bad = y.copy()
    y_bad[:, 5, :] = 1e6  # garbage on a masked node must not matter
    a, b = gcn.GCN(4, hidden=8, seed=3), gcn.GCN(4, hidden=8, seed=3)
    la, ga = a.loss_and_grads(ahat, x, y, mask, lam=0.0, delta=15.0)
    lb, gb = b.loss_and_grads(ahat, x, y_bad, mask, lam=0.0, delta=15.0)
    assert la == pytest.approx(lb)
    assert all(np.allclose(ga[k], gb[k]) for k in ga)


def test_save_load_roundtrip_and_transfer_to_a_larger_graph(tmp_path):
    ahat, x, y, ones = _toy(b=48)
    scaler = gcn.Standardizer().fit(x)
    model = gcn.GCN(4, hidden=8, seed=2)
    model.fit(ahat, x, y, ones, ones, val=None, scaler=scaler, epochs=5, batch=24, lr=1e-2, patience=5, delta=15.0, seed=2)
    names = [f"f{i}" for i in range(4)]
    nodes = [f"N{i}" for i in range(12)]
    model.save(tmp_path / "gcn.npz", feature_names=names, nodes=nodes)
    gcn.save_operator(tmp_path / "adj.npz", ahat)
    loaded, meta = gcn.GCN.load(tmp_path / "gcn.npz")
    assert meta["feature_names"] == names and meta["nodes"] == nodes
    assert np.allclose(loaded.predict(gcn.load_operator(tmp_path / "adj.npz"), x), model.predict(ahat, x))
    # no node-id features: the same weights run on a graph with a different number of nodes
    big = gcn.operator_inflow(ring_flows(30))
    out = loaded.predict(big, np.random.default_rng(0).normal(size=(2, 30, 4)))
    assert out.shape == (2, 30, 2) and np.isfinite(out).all()


def test_predict_accepts_a_single_snapshot():
    ahat, x, y, ones = _toy(b=4)
    model = gcn.GCN(4, hidden=8, seed=0)
    assert model.predict(ahat, x[0]).shape == (12, 2)


def test_features_that_never_varied_in_training_are_ignored_at_inference(tmp_path):
    ahat, x, y, ones = _toy(b=48)
    x[:, :, 3] = 0.0  # e.g. a zone one-hot for a climate absent from the training data
    model = gcn.GCN(4, hidden=8, seed=2)
    model.fit(ahat, x, y, ones, ones, val=None, scaler=gcn.Standardizer().fit(x), epochs=10, batch=24, lr=1e-2, patience=5, delta=15.0, seed=2)
    x_new = x.copy()
    x_new[:, :, 3] = 1.0
    assert np.allclose(model.predict(ahat, x), model.predict(ahat, x_new)), "an unseen level must not move the prediction"
    assert np.isfinite(model.predict(ahat, x_new)).all()
    model.save(tmp_path / "m.npz", feature_names=list("abcd"))
    again, _ = gcn.GCN.load(tmp_path / "m.npz")
    assert np.allclose(again.predict(ahat, x_new), model.predict(ahat, x))

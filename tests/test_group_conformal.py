"""Group-conformal calibration (re-implementation of Bhargav Teja & Divya Meena 2026): building blocks and invariants.
Agreement with the published numbers is checked separately (it needs the v4 data file)."""
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from src.data import TARGET, set_masks, wavelength_columns, wavelengths
from src.group_conformal import (GroupConformal, GroupConformalBC, TruncPLS, astm_group_interval, fold_assignment,
                                 group_metrics, group_weights, interval_score, out_of_group_models, split_conformal,
                                 weighted_quantile)
from src.preprocess import preprocess
from src.uncertainty import SpectralOOD


@pytest.fixture(scope='module')
def cal_subset(df):
    X, _ = preprocess(df[wavelength_columns(df)].to_numpy(), wavelengths(df))
    cal = set_masks(df)['Cal']
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(cal), 1500, replace=False)
    test = rng.choice(np.flatnonzero(set_masks(df)['Val Ext']), 300, replace=False)
    y = df[TARGET].to_numpy()
    return X[idx], y[idx], X[test]


@pytest.mark.parametrize('A', [1, 5, 12])
def test_truncated_pls_equals_refit(cal_subset, A):
    X, y, Xt = cal_subset
    trunc = TruncPLS(X, y, a_max=12)
    refit = PLSRegression(n_components=A, scale=False).fit(X, y)
    np.testing.assert_allclose(trunc.predict(Xt, A), refit.predict(Xt).ravel(), rtol=1e-8, atol=1e-8)
    ood = SpectralOOD(refit, X)   # T2 and Q of the project's OOD module, on the refitted A-component model
    np.testing.assert_allclose(trunc.t2(Xt, A), ood.hotelling_t2(Xt), rtol=1e-6)
    np.testing.assert_allclose(trunc.q(Xt, A), ood.q_residual(Xt), rtol=1e-6)


def test_group_weights_and_weighted_quantile():
    w = group_weights(['a', 'a', 'b'])
    np.testing.assert_allclose(w, [0.25, 0.25, 0.5])
    s, w = [4, 1, 3, 2], [0.4, 0.1, 0.3, 0.2]   # sorted: 1 (0.1), 2 (0.3), 3 (0.6), 4 (1.0)
    assert weighted_quantile(s, w, 0.3) == 2
    assert weighted_quantile(s, w, 0.31) == 3
    assert weighted_quantile(s, w, 1.0) == 4


def test_interval_score_hand_computed():
    lo, hi, alpha = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), 0.1
    np.testing.assert_allclose(interval_score(lo, hi, np.array([0.5, -1.0, 3.0]), alpha), [1.0, 1 + 20 * 1, 1 + 20 * 2])


def test_fold_assignment_one_group_per_fold():
    f = fold_assignment(['s1', 's2', 's1', 's3'])
    assert len(set(f)) == 3 and f[0] == f[2]


def test_out_of_group_models_drop_shared_units():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5))
    y = X @ np.arange(1, 6) + rng.normal(size=12)
    groups = np.repeat([0, 1, 2], 4)
    units = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10])  # unit 0 is measured in groups 0 and 1
    models = out_of_group_models(X, y, groups, units, a_max=3)
    # model for group 0: groups 1 and 2 (8 spectra) minus the spectrum of unit 0 in group 1 -> 7
    assert models[0].n_train == 7
    assert models[2].n_train == 8


def make_grouped(rng, n_groups=8, per_group=150, p=40, bias_sd=0.0):
    """Synthetic spectra: y depends on 3 latent factors; each group adds a bias to y that the spectra do not show."""
    B = rng.normal(size=(3, p))
    rows = []
    for g in range(n_groups):
        Z = rng.normal(size=(per_group, 3))
        X = Z @ B + 0.1 * rng.normal(size=(per_group, p))
        y = Z @ np.array([1.0, -0.5, 0.3]) + rng.normal(0.0, bias_sd) + 0.3 * rng.normal(size=per_group)
        rows.append((X, y, np.full(per_group, g)))
    X, y, g = (np.concatenate(v) for v in zip(*rows))
    return X, y, g, np.arange(len(y))   # one spectrum per unit


def split_groups(X, y, g, u, n_train_groups):
    tr = g < n_train_groups
    return (X[tr], y[tr], g[tr], u[tr]), (X[~tr], y[~tr], g[~tr], u[~tr])


def test_gcd_exchangeable_coverage_and_components():
    # same spectral model for training and new groups (one call), groups 0-7 train, 8-11 new
    (X, y, g, u), (Xt, yt, gt, _) = split_groups(*make_grouped(np.random.default_rng(1), n_groups=12), 8)
    gc = GroupConformal(alpha=0.1, a_max=8).fit(X, y, g, u)
    assert 3 <= gc.A <= 8 and len(gc.rmse_by_A) == 8
    _, lo, hi = gc.interval(Xt)
    assert 0.86 <= np.mean((yt >= lo) & (yt <= hi)) <= 0.94


def test_out_of_group_calibration_covers_group_bias():
    """With a group bias invisible in the spectra, out-of-group residuals contain it, so new groups stay covered;
    a random unit-wise split mostly does too here (the bias is in every training residual), which is why the method
    is checked on the published mango numbers rather than on synthetic data."""
    (X, y, g, u), (Xt, yt, gt, _) = split_groups(*make_grouped(np.random.default_rng(3), n_groups=24, bias_sd=0.8), 12)
    gc = GroupConformal(alpha=0.1, a_max=6).fit(X, y, g, u)
    _, lo, hi = gc.interval(Xt)
    assert group_metrics(yt, lo, hi, gt)['coverage'] > 0.8
    # ASTM-G and SCP baselines run and give finite intervals
    _, alo, ahi = astm_group_interval(gc, Xt)
    assert np.all(ahi > alo)
    assert np.isfinite(split_conformal(X, y, u, gc.A))


def test_bias_correction_variant_and_shift():
    rng = np.random.default_rng(5)
    X, y, g, u = make_grouped(rng, n_groups=8, bias_sd=0.8)
    u = np.arange(len(y)) // 2   # two spectra per unit
    gc = GroupConformal(alpha=0.1, a_max=6).fit(X, y, g, u)
    bc = GroupConformalBC(k=3, n_draws=5, alpha=0.1, a_max=6).fit(X, y, g, u)
    assert bc.A == gc.A and np.isfinite(bc.qhat)
    yhat, lo, hi = gc.interval(X[:5])
    yhat2, lo2, hi2 = gc.interval(X[:5], shift=0.7)
    np.testing.assert_allclose(yhat2 - yhat, 0.7)
    np.testing.assert_allclose(hi2 - lo2, hi - lo)


def irls_poisson(F, y, n_iter=100):
    """Poisson log-link GLM by iteratively reweighted least squares, written out by hand (independent of sklearn)."""
    Z = np.column_stack([np.ones(len(y)), F])
    beta = np.r_[np.log(y.mean()), np.zeros(F.shape[1])]
    for _ in range(n_iter):
        mu = np.exp(Z @ beta)
        z = Z @ beta + (y - mu) / mu
        beta_new = np.linalg.solve(Z.T @ (mu[:, None] * Z), Z.T @ (mu * z))
        if np.max(np.abs(beta_new - beta)) < 1e-12:
            return beta_new
        beta = beta_new
    return beta


def test_scale_model_matches_hand_irls_and_recovers_coefficients():
    from src.group_conformal import ScaleModel
    rng = np.random.default_rng(6)
    t2 = rng.gamma(2.0, 3.0, size=4000)
    q = rng.gamma(2.0, 1.0, size=4000)
    q_med = float(np.median(q))
    true = np.array([-0.5, 0.3, 0.6])
    mean_abs = np.exp(true[0] + true[1] * np.log1p(t2) + true[2] * np.log(q / q_med))
    abs_r = mean_abs * np.abs(rng.normal(size=4000)) / np.sqrt(2 / np.pi)   # E|r| = mean_abs
    sm = ScaleModel(q_med).fit(abs_r, t2, q)
    np.testing.assert_allclose(sm.coef, irls_poisson(sm.features(t2, q), abs_r), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(sm.coef, true, atol=0.06)
    assert sm(t2, q).min() >= sm.floor > 0


def reference_gcd(X, y, groups, units, Xt, alpha, a_max):
    """GC-D written directly from Section 2.3 with a separate code path: PLS refitted for every A and fold,
    T2/Q from the project's SpectralOOD, GLM by hand IRLS, Eq. (4) by an explicit search."""
    labels = np.unique(groups)
    w = np.array([1.0 / (len(labels) * np.sum(groups == g)) for g in groups])

    def fold_fit(g, A):
        held = groups == g
        train = ~held & ~np.isin(units, units[held])
        return held, PLSRegression(n_components=A, scale=False).fit(X[train], y[train]), X[train]

    rmse = []
    for A in range(1, a_max + 1):
        r = np.empty(len(y))
        for g in labels:
            held, m, _ = fold_fit(g, A)
            r[held] = y[held] - m.predict(X[held]).ravel()
        rmse.append(np.sqrt(np.sum(w * r ** 2)))
    A = int(np.argmin(rmse)) + 1
    r, t2, q = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    for g in labels:
        held, m, Xtr = fold_fit(g, A)
        ood = SpectralOOD(m, Xtr)
        r[held] = y[held] - m.predict(X[held]).ravel()
        t2[held], q[held] = ood.hotelling_t2(X[held]), ood.q_residual(X[held])
    final = PLSRegression(n_components=A, scale=False).fit(X, y)
    final_ood = SpectralOOD(final, X)
    q_med = np.median(final_ood.q_residual(X))
    feats = lambda t2_, q_: np.column_stack([np.log1p(t2_), np.log(q_ / q_med)])
    beta = irls_poisson(feats(t2, q), np.abs(r))
    sig = lambda t2_, q_: np.exp(beta[0] + feats(t2_, q_) @ beta[1:])
    floor = 0.1 * np.median(sig(t2, q))
    s = np.abs(r) / np.maximum(sig(t2, q), floor)
    qhat = next(t for t in np.sort(s) if w[s <= t].sum() >= 1 - alpha - 1e-12)
    yhat = final.predict(Xt).ravel()
    half = qhat * np.maximum(sig(final_ood.hotelling_t2(Xt), final_ood.q_residual(Xt)), floor)
    return A, qhat, yhat - half, yhat + half


def test_gcd_matches_independent_reference():
    rng = np.random.default_rng(7)
    X, y, g, _ = make_grouped(rng, n_groups=6, per_group=40, p=15, bias_sd=0.5)
    u = np.arange(len(y))
    u[45] = u[3]          # one unit measured in groups 0 and 1, one in groups 2 and 4
    u[170] = u[90]
    keep = ~((g == 1) & (np.arange(len(y)) >= 55))   # group 1 keeps 15 of 40 spectra: unequal group sizes, so
    X, y, g, u = X[keep], y[keep], g[keep], u[keep]   # group-balanced and uniform weights differ
    (Xtr, ytr, gtr, utr), (Xt, _, _, _) = split_groups(X, y, g, u, 5)
    A, qhat, lo, hi = reference_gcd(Xtr, ytr, gtr, utr, Xt, alpha=0.1, a_max=6)
    gc = GroupConformal(alpha=0.1, a_max=6).fit(Xtr, ytr, gtr, utr)
    assert gc.A == A
    assert gc.qhat == pytest.approx(qhat, rel=1e-8)
    _, lo2, hi2 = gc.interval(Xt)
    np.testing.assert_allclose(lo2, lo, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(hi2, hi, rtol=1e-7, atol=1e-9)

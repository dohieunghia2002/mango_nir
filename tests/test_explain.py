"""M5 explanations (Step 4a): exact properties of the band contributions, the deletions and the width decomposition."""
import numpy as np
import pytest
from scipy.signal import savgol_filter

from src.explain import (band_slices, band_sums, delete_interp, delete_mean, deletion_test, linear_contributions,
                         merged_ranges, sigma_components, stability, unit_means)
from src.group_conformal import GroupConformal, TruncPLS

WL_RAW = np.arange(285, 1201, 3)
WL_TRIM = WL_RAW[(WL_RAW >= 684) & (WL_RAW <= 990)]


def synthetic(n=300, p=103, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) @ rng.normal(size=(p, p)) * 0.1
    y = X[:, :5].sum(axis=1) + 0.1 * rng.normal(size=n)
    return X, y


def test_bands_cover_the_window():
    bands = band_slices(len(WL_TRIM))
    assert len(WL_TRIM) == 103 and len(bands) == 10
    assert bands[0][0] == 0 and bands[-1][1] == 103
    assert all(b[1] == c[0] for b, c in zip(bands, bands[1:]))        # contiguous
    assert [b - a for a, b in bands] == [10] * 9 + [13]


def test_contributions_sum_to_prediction_and_mean_deletion_is_exact():
    X, y = synthetic()
    m = TruncPLS(X, y, a_max=8)
    C = linear_contributions(m, X, 8)
    np.testing.assert_allclose(m.y_mean + C.sum(axis=1), m.predict(X, 8), rtol=1e-10, atol=1e-10)
    bands = band_slices(X.shape[1])
    B = band_sums(C, bands)
    Xp = delete_mean(X, [1, 4], bands, m.x_mean)
    # the known-in-advance property of the sanity check: the change equals minus the deleted band contributions
    np.testing.assert_allclose(m.predict(Xp, 8) - m.predict(X, 8), -(B[:, 1] + B[:, 4]), rtol=1e-10, atol=1e-10)


def test_interp_deletion_is_linear_inside_and_unchanged_outside():
    rng = np.random.default_rng(1)
    raw = np.cumsum(rng.normal(size=(4, len(WL_RAW))), axis=1)
    bands = band_slices(len(WL_TRIM))
    assert merged_ranges([3, 2, 7], bands, WL_TRIM) == [(WL_TRIM[20], WL_TRIM[39]), (WL_TRIM[70], WL_TRIM[79])]
    out = delete_interp(raw, WL_RAW, [2, 3], bands, WL_TRIM)
    inside = (WL_RAW >= WL_TRIM[20]) & (WL_RAW <= WL_TRIM[39])
    np.testing.assert_array_equal(out[:, ~inside], raw[:, ~inside])
    seg = out[:, np.flatnonzero(inside)[0] - 1: np.flatnonzero(inside)[-1] + 2]
    np.testing.assert_allclose(np.diff(seg, 2, axis=1), 0, atol=1e-10)       # straight line incl. both anchors
    d2 = savgol_filter(out, 17, 2, deriv=2, axis=1)
    core = np.flatnonzero(inside)[8:-8]                                        # SG window 17: 8 points each side
    np.testing.assert_allclose(d2[:, core], 0, atol=1e-10)


def test_sigma_components_reconstruct_log_sigma():
    rng = np.random.default_rng(2)
    X, y = synthetic(n=400)
    g = np.repeat(np.arange(4), 100)
    y = y + 0.5 * rng.normal(size=4)[g]
    gc = GroupConformal(alpha=0.1, a_max=6, level='unit').fit(X, y, g, np.arange(len(y)))
    t2, q = gc.final.t2(X[:20], gc.A), gc.final.q(X[:20], gc.A)
    comp = sigma_components(gc, t2, q)
    b0, b1, b2 = gc.scale_model.coef
    np.testing.assert_allclose(comp['log_sigma_typical'] + comp['from_T2'] + comp['from_Q'],
                               b0 + b1 * np.log1p(t2) + b2 * np.log(q / gc.q_med), rtol=1e-12)


def test_unit_means_and_stability_of_identical_contributions():
    ids, mu = unit_means(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), [7, 7, 9])
    np.testing.assert_array_equal(ids, [7, 9])
    np.testing.assert_allclose(mu, [[2.0, 3.0], [5.0, 6.0]])
    rng = np.random.default_rng(3)
    per_unit = rng.normal(size=(20, 10))
    units = np.repeat(np.arange(20), 2)
    st = stability(per_unit[units], units, units // 10)
    assert st['spearman_same_median'] == pytest.approx(1.0) and st['jaccard_top_same_mean'] == 1.0
    assert st['n_same_pairs'] == 20


def test_deletion_test_with_mean_strategy():
    X, y = synthetic(n=200)
    m = TruncPLS(X, y, a_max=8)
    bands = band_slices(X.shape[1])
    units = np.arange(len(y)) // 2

    def decider(Xr, ur):
        ids, c = unit_means(m.predict(Xr, 8), ur)
        return ids, c, np.full(len(ids), 0.3)

    base = decider(X, units)
    ids, Bu = unit_means(band_sums(linear_contributions(m, X, 8), bands), units)
    ranking = {u: list(np.argsort(-np.abs(Bu[i]))) for i, u in enumerate(ids)}
    rows = deletion_test(decider, None, X, units, None, None, bands, ranking, None, m.x_mean, base, [float(np.median(y))],
                         'mean', n_draws=20)
    top = {r['m']: r for r in rows if r['order'] == 'top'}
    rnd = {r['m']: r for r in rows if r['order'] == 'random'}
    assert top[1]['abs_change'] >= rnd[1]['abs_change']           # the largest band beats a random one, per unit
    assert top[10]['abs_change'] == pytest.approx(rnd[10]['abs_change'])   # all bands deleted: order is irrelevant


def test_block_bands_and_index_deletion():
    from src.explain import band_index, block_bands
    bands = band_slices(103)
    bb = block_bands(bands, 6, 103)
    assert len(bb) == 10 and sum(len(b) for b in bb) == 618
    assert np.array_equal(np.sort(np.concatenate(bb)), np.arange(618))           # a partition of the 618 features
    np.testing.assert_array_equal(bb[0][:10], np.arange(10))
    np.testing.assert_array_equal(bb[0][10:20], np.arange(103, 113))
    np.testing.assert_array_equal(band_index(bands[9]), np.arange(90, 103))
    X = np.random.default_rng(4).normal(size=(5, 618)).astype(np.float32)
    Xp = delete_mean(X, [0], bb, np.zeros(618, dtype=np.float32))
    assert Xp.dtype == np.float32 and np.all(Xp[:, bb[0]] == 0) and np.array_equal(Xp[:, bb[1]], X[:, bb[1]])


def test_integrated_gradients_on_known_functions():
    from src.explain_nn import completeness_error, gradient_x_input, integrated_gradients
    rng = np.random.default_rng(5)
    w = rng.normal(size=8)
    X = rng.normal(size=(6, 8)).astype(np.float32)
    base = np.zeros(8, dtype=np.float32)
    # linear F = w.x: IG = GxI = w * (x - b) exactly
    lin = lambda Z: np.tile(w, (len(Z), 1))
    np.testing.assert_allclose(integrated_gradients(lin, X, base, steps=4), w * X, rtol=1e-6)
    np.testing.assert_allclose(gradient_x_input(lin, X, base), w * X, rtol=1e-6)
    # F = sum w x^2 (gradient 2 w x): IG_j = w_j x_j^2 exactly with the midpoint rule (integrand linear in alpha);
    # GxI = 2 w x^2 differs by a factor 2, so the two methods are distinguishable
    quad = lambda Z: 2 * w * Z
    ig = integrated_gradients(quad, X, base, steps=8)
    np.testing.assert_allclose(ig, w * X.astype(np.float64) ** 2, rtol=1e-5)
    err, rel = completeness_error(ig, (w * X.astype(np.float64) ** 2).sum(axis=1), 0.0)
    assert rel.max() < 1e-5
    # nonlinear F = sum sin(x): completeness error shrinks with more steps
    f = lambda Z: np.sin(Z).sum(axis=1)
    g = lambda Z: np.cos(Z)
    e8 = completeness_error(integrated_gradients(g, X, base, steps=8), f(X), f(base[None]))[0].max()
    e64 = completeness_error(integrated_gradients(g, X, base, steps=64), f(X), f(base[None]))[0].max()
    assert e64 < e8 and e64 < 1e-3

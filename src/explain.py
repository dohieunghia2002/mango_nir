"""M5 - explanations of harvest decisions and checks of their faithfulness (Step 4a: PLS).

Design fixed on 26/11/2026 before any result (agreed with the user):
- explained layer: GC-D with season groups on the common PLS model (11 LV, Cal + Tuning), fruit-level calibration
  (src/step2_gcd_baseline.py, point_model 'A=11'); the Step 2 layer (constant half-width) as a secondary analysis;
- explanation of each fruit's decision, all quantitative:
  (a) margin p - T and its ratio to the half-width;
  (b) half-width and why: log sigma = b0 + b1 log(1 + T2) + b2 log(Q / Q_med) splits into a T2 part (extrapolation in the
      score space) and a Q part (spectral variation outside the model), relative to a typical calibration fruit;
  (c) flags: fruit-mean Q or T2 above the 99 % quantile of the calibration fruit;
  (d) band contributions: for a linear PLS model, contribution_j = b_j (x_j - xbar_j), summed over 10 fixed contiguous
      bands of the 684-990 nm window (10 points each, the last band 13 points); no chemical meaning is assigned;
- faithfulness: delete the m top-ranked bands (|contribution|) vs m random bands (100 draws; in each draw one random
  band order shared by all fruit), m = 1..10; outcome = |change of the fruit prediction| and share of fruit whose
  decision changes (the full decision is recomputed: centre and half-width). Two deletions:
    'mean'   - feature values of the band replaced by the training mean (sanity check: for PLS this is the implicit
               baseline of the contributions, so the result is known in advance),
    'interp' - raw absorbance linearly interpolated across the band, then preprocessed again (main test: a physically
               plausible spectrum, not the baseline of the explanation);
- stability: band contributions of the spectra of one fruit vs spectra of different fruit of the same population
  (Spearman correlation of the signed band contributions, Jaccard overlap of the 3 bands of largest |contribution|).
"""
import numpy as np
from scipy import stats

N_BANDS = 10


def band_slices(n_vars, n_bands=N_BANDS):
    """Contiguous index ranges: n_bands bands of n_vars // n_bands variables, the last band takes the remainder."""
    width = n_vars // n_bands
    starts = [i * width for i in range(n_bands)]
    stops = starts[1:] + [n_vars]
    return list(zip(starts, stops))


def band_labels(wl, bands):
    return [f'{wl[a]}-{wl[b - 1]} nm' for a, b in bands]


def band_index(band):
    """Column indices of a band: a (start, stop) pair of a contiguous band, or an index array (e.g. block_bands)."""
    if isinstance(band, np.ndarray):
        return band
    a, b = band
    return np.arange(a, b)


def block_bands(bands, n_blocks, block_size):
    """Bands over features made of n_blocks concatenated pretreatments of the same wavelengths (the 'AT' input of the
    CNN: 6 blocks x 103 wavelengths): band j takes its wavelengths in every block."""
    return [np.concatenate([np.arange(a, b) + k * block_size for k in range(n_blocks)]) for a, b in bands]


def linear_contributions(model, X, A):
    """Per-variable contributions of a linear PLS model (TruncPLS): b_j (x_j - xbar_j); they sum, with the training
    mean of y, to the prediction."""
    b = model.R[:, :A] @ model.qy[:A]
    return (np.asarray(X, dtype=float) - model.x_mean) * b


def band_sums(C, bands):
    return np.column_stack([C[:, band_index(b)].sum(axis=1) for b in bands])


def unit_means(values, units):
    """(unit ids, per-unit mean of the rows of values), units in sorted order as np.unique."""
    ids, inv, n = np.unique(np.asarray(units), return_inverse=True, return_counts=True)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return ids, np.bincount(inv, weights=values) / n
    return ids, np.column_stack([np.bincount(inv, weights=values[:, j]) / n for j in range(values.shape[1])])


def sigma_components(gc, t2, q):
    """Split log sigma of a fitted unit-level GroupConformal into a T2 part and a Q part relative to a typical
    calibration fruit (median calibration T2; Q_med, where log(Q / Q_med) = 0). Also whether the floor applies."""
    b0, b1, b2 = gc.scale_model.coef
    t2_ref = float(np.median(gc.u_t2))
    raw_log = b0 + b1 * np.log1p(t2) + b2 * np.log(q / gc.q_med)
    return {'log_sigma_typical': b0 + b1 * np.log1p(t2_ref), 'from_T2': b1 * (np.log1p(t2) - np.log1p(t2_ref)),
            'from_Q': b2 * np.log(q / gc.q_med), 'floored': np.exp(raw_log) < gc.scale_model.floor, 't2_ref': t2_ref}


def spectral_flags(gc, t2, q, level=0.99):
    """Fruit-mean Q or T2 above the given quantile of the calibration fruit (unit-level calibration values)."""
    return {'flag_Q': q > np.quantile(gc.u_q, level), 'flag_T2': t2 > np.quantile(gc.u_t2, level)}


def delete_mean(X, band_idx, bands, x_mean):
    """Feature values of the listed bands replaced by the training mean (same bands for every row)."""
    Xp = np.array(X, copy=True)
    for i in band_idx:
        idx = band_index(bands[i])
        Xp[:, idx] = x_mean[idx]
    return Xp


def delete_interp(raw, wl_raw, band_idx, bands, wl_trim):
    """Raw absorbance linearly interpolated across the wavelength range of the listed bands (adjacent bands merged),
    between the nearest raw wavelengths outside the range; the caller preprocesses the result again."""
    raw_p = np.array(raw, dtype=float, copy=True)
    wl_raw = np.asarray(wl_raw)
    for lo, hi in merged_ranges(band_idx, bands, wl_trim):
        inside = np.flatnonzero((wl_raw >= lo) & (wl_raw <= hi))
        left, right = inside[0] - 1, inside[-1] + 1
        frac = (wl_raw[inside] - wl_raw[left]) / (wl_raw[right] - wl_raw[left])
        raw_p[:, inside] = raw_p[:, [left]] + (raw_p[:, [right]] - raw_p[:, [left]]) * frac
    return raw_p


def merged_ranges(band_idx, bands, wl_trim):
    """Wavelength ranges (lo, hi) of the listed bands, adjacent bands merged into one range."""
    out = []
    for i in sorted(band_idx):
        lo, hi = wl_trim[bands[i][0]], wl_trim[bands[i][1] - 1]
        if out and i - 1 in band_idx and out[-1][2] == i - 1:
            out[-1] = (out[-1][0], hi, i)
        else:
            out.append((lo, hi, i))
    return [(lo, hi) for lo, hi, _ in out]


def deletion_test(decider, raw, X, units, wl_raw, wl_trim, bands, ranking, featurize, x_mean, base, T_values,
                  strategy, n_draws=100, seed=0):
    """Deletion test for one decision layer.
    decider(X_rows, unit_rows) -> (unit ids, centre, half); base = decider(X, units) on the unperturbed spectra;
    ranking: dict unit id -> band indices ordered by decreasing |contribution|; featurize(raw) -> features.
    Returns rows per m and order ('top' or 'random'): mean |change of centre|, and per T the share of units whose
    decision (ready / not ready / defer) changes; for 'random' the mean over draws and the share of draws whose
    decision-change share is at least the top-ranked one (a permutation-style p-value)."""
    from src.decision import decide
    ids, centre0, half0 = base
    pos = {u: i for i, u in enumerate(ids)}
    units = np.asarray(units)
    n_b = len(bands)

    def perturbed(rows, band_idx):
        if strategy == 'mean':
            return delete_mean(X[rows], band_idx, bands, x_mean)
        return featurize(delete_interp(raw[rows], wl_raw, band_idx, bands, wl_trim))

    def outcome(centre, half):
        out = {'abs_change': float(np.mean(np.abs(centre - centre0)))}
        for T in T_values:
            out[f'decision_change_T{T:g}'] = float(np.mean(decide(centre, half, T) != decide(centre0, half0, T)))
        return out

    rng = np.random.default_rng(seed)
    orders = [rng.permutation(n_b) for _ in range(n_draws)]
    rows_out = []
    row_pos = np.array([pos[u] for u in units])
    for m in range(1, n_b + 1):
        if strategy == 'mean':
            # top-ranked, all units at once: each row gets the m top bands of its own unit (a unit's decision depends
            # only on its own rows, so this equals deleting unit by unit)
            in_top = np.array([[j in set(ranking[u][:m]) for j in range(n_b)] for u in ids])   # units x bands
            Xp = np.array(X, copy=True)
            for j in range(n_b):
                idx = band_index(bands[j])
                rows = np.flatnonzero(in_top[row_pos, j])
                Xp[np.ix_(rows, idx)] = x_mean[idx]
            uid, centre, half = decider(Xp, units)
            assert np.array_equal(uid, ids)
        else:
            # top-ranked: units grouped by their set of m top bands (interpolation is done per band set)
            centre, half = np.empty(len(ids)), np.empty(len(ids))
            sets = {}
            for u in ids:
                sets.setdefault(tuple(sorted(ranking[u][:m])), []).append(u)
            for band_set, us in sets.items():
                rows = np.flatnonzero(np.isin(units, us))
                uid, c, h = decider(perturbed(rows, list(band_set)), units[rows])
                idx = [pos[x] for x in uid]
                centre[idx], half[idx] = c, h
        top = outcome(centre, half)
        rows_out.append({'strategy': strategy, 'm': m, 'order': 'top', **top})
        # random: one band order per draw, shared by all units
        draws = []
        for order in orders:
            uid, c, h = decider(perturbed(np.arange(len(units)), list(order[:m])), units)
            assert np.array_equal(uid, ids)
            draws.append(outcome(c, h))
        rnd = {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}
        p_values = {f'p_{k}': float(np.mean([d[k] >= top[k] for d in draws])) for k in draws[0]}
        rows_out.append({'strategy': strategy, 'm': m, 'order': 'random', **rnd, **p_values})
    return rows_out


def stability(band_contrib, units, pops, top=3, seed=0):
    """Agreement of the band contributions between spectra of the same fruit, and between spectra of different fruit
    of the same population (one random partner per same-fruit pair). Spearman correlation of the signed band
    contributions and Jaccard overlap of the `top` bands of largest |contribution|."""
    units, pops = np.asarray(units), np.asarray(pops)
    rng = np.random.default_rng(seed)

    def pair_stats(i, j):
        rho = stats.spearmanr(band_contrib[i], band_contrib[j])[0]
        ti = set(np.argsort(-np.abs(band_contrib[i]))[:top])
        tj = set(np.argsort(-np.abs(band_contrib[j]))[:top])
        return rho, len(ti & tj) / len(ti | tj)

    same, other = [], []
    for u in np.unique(units):
        rows = np.flatnonzero(units == u)
        partners = np.flatnonzero((pops == pops[rows[0]]) & (units != u))
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                same.append(pair_stats(rows[a], rows[b]))
                if len(partners):
                    other.append(pair_stats(rows[a], rng.choice(partners)))
    same, other = np.array(same), np.array(other)
    return {'n_same_pairs': int(len(same)), 'n_other_pairs': int(len(other)),
            'spearman_same_median': float(np.median(same[:, 0])), 'spearman_other_median': float(np.median(other[:, 0])),
            'jaccard_top_same_mean': float(np.mean(same[:, 1])), 'jaccard_top_other_mean': float(np.mean(other[:, 1])),
            'top': top}


def explanation_text(rec):
    """Short Vietnamese explanation of one fruit's decision from its record (for the demo)."""
    verdict = {'ready': 'đủ chín', 'not_ready': 'chưa chín', 'defer': 'chưa quyết'}[rec['decision']]
    parts = [f"Quyết định: {verdict}. DMC dự đoán {rec['p']:.1f} %FW, ngưỡng {rec['T']:.0f} %FW "
             f"(cách ngưỡng {rec['margin']:+.1f}), khoảng 90% ±{rec['half']:.1f}."]
    if rec['decision'] == 'defer':
        parts.append('Ngưỡng nằm trong khoảng bất định nên chưa đủ căn cứ; cần đo phá mẫu hoặc chờ thêm.')
    widen = [n for n, k in (('phổ nằm xa vùng hiệu chỉnh (T²)', 'from_T2'), ('phổ có biến động mô hình chưa giải thích (Q)', 'from_Q'))
             if rec[k] > np.log(1.25)]
    if widen:
        parts.append('Khoảng rộng hơn quả điển hình vì ' + ' và '.join(widen) + '.')
    if rec.get('flag_Q') or rec.get('flag_T2'):
        parts.append('Cảnh báo: phổ bất thường so với dữ liệu hiệu chỉnh.')
    parts.append('Dải bước sóng đóng góp nhiều nhất: ' + ', '.join(rec['top_bands']) + '.')
    return ' '.join(parts)

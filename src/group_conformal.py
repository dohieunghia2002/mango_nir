"""Group-conformal calibration with PLS diagnostics (GC-D) and its bias-corrected variant (GC-D-BC), re-implemented from
the description in Bhargav Teja & Divya Meena (2026), Talanta 314:130941, doi:10.1016/j.talanta.2026.130941:
Section 2.3 (method), Section 4.8 and Supplementary Table S22 (GC-D-BC), Supplementary Table S1 (settings). The
article's code is not in its supplementary material, so choices the paper leaves open are marked ASSUMPTION; they are
checked by reproducing published numbers before the method is used in this project.

GC-D, for calibration spectra x_i with reference y_i, calibration group g_i (season, instrument, population) and physical
unit u_i (fruit):
  1. out-of-group residuals r_i = y_i - f_{-k(i)}(x_i) from PLS models fitted without fold k(i) and without every unit
     that also occurs in it, with the diagnostics T2_i and Q_i of the same model; the number of latent variables A
     minimises the group-balanced RMSE of these residuals;
  2. scale model log sigma = b0 + b1 log(1 + T2) + b2 log(Q / Q_med), fitted by a log-link GLM with Poisson
     quasi-likelihood on |r_i|; sigma is floored at 10 % of its median; Q_med is the median training Q of the final model;
  3. q = weighted quantile of s_i = |r_i| / sigma_i with group-balanced weights 1 / (K N_g);
  4. interval yhat(x) -/+ q sigma(x), with the final PLS model fitted on all calibration spectra.

Responsibilities of the caller (not checked here): the calibration spectra passed to fit() must not contain any unit of
the held-out (test) group, and in GC-D-BC evaluation the k labelled units that give the shift must be left out of the
spectra whose coverage is scored.
"""
import warnings

import numpy as np
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

A_MAX = 20          # Table S1, mango
FLOOR_FRAC = 0.1    # Table S1, scale_model.floor_frac


class TruncPLS:
    """PLS1 fitted once with a_max components. For PLS1 (NIPALS with deflation) the first A components equal those of an
    A-component fit, so predictions, scores, T2 and Q for any A <= a_max come from the truncated rotations/loadings."""

    def __init__(self, X, y, a_max=A_MAX):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.a_max = int(min(a_max, X.shape[1], X.shape[0] - 1))
        pls = PLSRegression(n_components=self.a_max, scale=False).fit(X, y)
        self.x_mean = X.mean(axis=0)
        self.y_mean = float(y.mean())
        self.R = pls.x_rotations_          # p x a_max, scores t = (x - x_mean) R
        self.P = pls.x_loadings_           # p x a_max
        self.qy = pls.y_loadings_.ravel()  # a_max
        self.n_train = X.shape[0]
        T = (X - self.x_mean) @ self.R
        self.score_var = T.var(axis=0, ddof=1)       # s_a^2 of Eq. (2)
        self.score_ss = (T ** 2).sum(axis=0)         # t_a' t_a, for the leverage of classical intervals

    def scores(self, X, A):
        return (np.asarray(X, dtype=float) - self.x_mean) @ self.R[:, :A]

    def predict(self, X, A):
        return self.y_mean + self.scores(X, A) @ self.qy[:A]

    def t2(self, X, A):
        return (self.scores(X, A) ** 2 / self.score_var[:A]).sum(axis=1)

    def q(self, X, A):
        Xc = np.asarray(X, dtype=float) - self.x_mean
        return ((Xc - self.scores(X, A) @ self.P[:, :A].T) ** 2).sum(axis=1)

    def leverage(self, X, A):
        return 1 / self.n_train + (self.scores(X, A) ** 2 / self.score_ss[:A]).sum(axis=1)


def group_weights(groups):
    """w_i = 1 / (K N_g): every group carries the same total weight; the weights sum to 1."""
    groups = np.asarray(groups)
    labels, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / (len(labels) * counts[inverse])


def weighted_quantile(scores, weights, level):
    """Eq. (4): inf{t : sum_i w_i 1(s_i <= t) >= level}, with weights normalised to sum 1."""
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(s, kind='stable')
    cum = np.cumsum(w[order]) / w.sum()
    return float(s[order][np.searchsorted(cum, level - 1e-12)])


def fold_assignment(groups, n_folds=None, rng=None):
    """Fold index per spectrum. ASSUMPTION: one calibration group per fold unless n_folds is given (the paper uses one
    group per fold 'if K is small' without a threshold); with n_folds, groups are assigned to folds at random."""
    labels, inverse = np.unique(np.asarray(groups), return_inverse=True)
    if n_folds is None or n_folds >= len(labels):
        return inverse
    rng = rng or np.random.default_rng(0)
    group_fold = rng.permutation(np.arange(len(labels)) % n_folds)
    return group_fold[inverse]


def out_of_group_models(X, y, folds, units=None, a_max=A_MAX):
    """One TruncPLS per fold, fitted on the other folds after removing every unit that occurs in the held-out fold."""
    models = {}
    for k in np.unique(folds):
        held = folds == k
        train = ~held
        if units is not None:
            train &= ~np.isin(units, np.unique(units[held]))
        models[k] = TruncPLS(X[train], y[train], a_max)
    return models


def oog_predictions(models, X, folds, A):
    pred = np.empty(len(X))
    for k, m in models.items():
        idx = folds == k
        pred[idx] = m.predict(X[idx], A)
    return pred


def select_components(models, X, y, folds, groups):
    """A minimising the group-balanced RMSE of the out-of-group residuals; returns (A, rmse per A)."""
    w = group_weights(groups)
    a_max = min(m.a_max for m in models.values())
    rmse = np.array([np.sqrt(np.sum(w * (y - oog_predictions(models, X, folds, A)) ** 2)) for A in range(1, a_max + 1)])
    return int(np.argmin(rmse)) + 1, rmse


def interval_score(lo, hi, y, alpha):
    """Eq. (6): Winkler interval score per spectrum."""
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


class ScaleModel:
    """sigma(x) = exp(b0 + b1 log(1 + T2) + b2 log(Q / Q_med)), Poisson quasi-likelihood GLM with log link on |r|.
    ASSUMPTION: unweighted GLM fit (the paper does not mention weights for this step); the Poisson point estimates are
    those of the quasi-likelihood. The floor is 10 % of the median sigma over the calibration spectra."""

    def __init__(self, q_med):
        self.q_med = q_med

    def features(self, t2, q):
        return np.column_stack([np.log1p(t2), np.log(np.maximum(q, 1e-300) / self.q_med)])

    def fit(self, abs_resid, t2, q):
        # Newton (Cholesky) solver: exact maximum-likelihood fit of the 3 coefficients, no approximate stopping
        self.glm = PoissonRegressor(alpha=0.0, solver='newton-cholesky', max_iter=200, tol=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)   # an error instead of a silent bad fit
            self.glm.fit(self.features(t2, q), abs_resid)
        self.coef = np.r_[self.glm.intercept_, self.glm.coef_]   # b0, b1, b2 of Eq. (3)
        self.floor = FLOOR_FRAC * float(np.median(self.glm.predict(self.features(t2, q))))
        return self

    def __call__(self, t2, q):
        return np.maximum(self.glm.predict(self.features(t2, q)), self.floor)


class GroupConformal:
    """GC-D (scale='diag', weights='group') and the ablation cells of Fig. 5: weights 'uniform' (U) or 'group' (W),
    scores 'abs' (A) or 'diag' (D); calibration folds are always out-of-group here (G).
    fit(...) then interval(X) -> (yhat, lo, hi)."""

    def __init__(self, alpha=0.1, a_max=A_MAX, n_components=None, scale='diag', weights='group', n_folds=None, seed=0):
        self.alpha, self.a_max, self.n_components = alpha, a_max, n_components
        self.scale, self.weights, self.n_folds, self.seed = scale, weights, n_folds, seed

    def fit(self, X, y, groups, units=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        groups = np.asarray(groups)
        units = None if units is None else np.asarray(units)
        self.folds = fold_assignment(groups, self.n_folds, np.random.default_rng(self.seed))
        models = out_of_group_models(X, y, self.folds, units, self.a_max)
        self.A, self.rmse_by_A = (select_components(models, X, y, self.folds, groups) if self.n_components is None
                                  else (self.n_components, None))
        A = self.A
        self.final = TruncPLS(X, y, max(A, 1) if self.n_components else self.a_max)
        self.resid = y - oog_predictions(models, X, self.folds, A)   # out-of-group residuals r_i
        self.t2_cal = np.empty(len(X))
        self.q_cal = np.empty(len(X))
        for k, m in models.items():
            idx = self.folds == k
            self.t2_cal[idx] = m.t2(X[idx], A)
            self.q_cal[idx] = m.q(X[idx], A)
        self.q_med = float(np.median(self.final.q(X, A)))
        self.groups, self.units = groups, units
        self._calibrate(np.abs(self.resid), self.t2_cal, self.q_cal, groups)
        return self

    def _calibrate(self, abs_resid, t2, q, groups):
        w = group_weights(groups) if self.weights == 'group' else np.full(len(abs_resid), 1.0 / len(abs_resid))
        if self.scale == 'diag':
            self.scale_model = ScaleModel(self.q_med).fit(abs_resid, t2, q)
            sigma = self.scale_model(t2, q)
        else:
            self.scale_model = None
            sigma = np.ones(len(abs_resid))
        self.qhat = weighted_quantile(abs_resid / sigma, w, 1 - self.alpha)

    def sigma(self, X):
        if self.scale_model is None:
            return np.ones(len(X))
        return self.scale_model(self.final.t2(X, self.A), self.final.q(X, self.A))

    def interval(self, X, shift=0.0):
        """Interval for new spectra; shift is added to the centre (bias correction: mean residual y - yhat of k labelled
        units of the new group)."""
        yhat = self.final.predict(X, self.A) + shift
        half = self.qhat * self.sigma(X)
        return yhat, yhat - half, yhat + half


class GroupConformalBC(GroupConformal):
    """GC-D-BC (Section 4.8, Table S22): the test-time correction (centre shifted by the mean residual of k labelled
    units of the new group) is mimicked in every calibration group before sigma and q are fitted.
    ASSUMPTIONS: the bias estimate is the mean residual over all spectra of the k drawn units; the corrected residuals of
    the remaining units are pooled over n_draws random draws per group; group-balanced weights are computed on the
    pooled rows, so every group still carries 1/K; the scale model is fitted on the pooled rows (unweighted)."""

    def __init__(self, k, n_draws=20, **kw):
        super().__init__(**kw)
        self.k, self.n_draws = k, n_draws

    def fit(self, X, y, groups, units):
        super().fit(X, y, groups, units)   # calibrates plain GC-D first, then recalibrates on corrected residuals
        rng = np.random.default_rng(self.seed)
        rows, resid, grp = [], [], []
        for g in np.unique(self.groups):
            in_g = np.flatnonzero(self.groups == g)
            g_units = np.unique(self.units[in_g])
            if len(g_units) <= self.k:
                continue
            for _ in range(self.n_draws):
                ref = np.isin(self.units[in_g], rng.choice(g_units, self.k, replace=False))
                bias = self.resid[in_g[ref]].mean()
                rest = in_g[~ref]
                rows.append(rest)
                resid.append(self.resid[rest] - bias)
                grp.append(np.full(len(rest), g))
        rows = np.concatenate(rows)
        self._calibrate(np.abs(np.concatenate(resid)), self.t2_cal[rows], self.q_cal[rows], np.concatenate(grp))
        return self


def split_conformal(X, y, units, A, alpha=0.1, frac=0.25, seed=0):
    """SCP baseline: random unit-wise split (calibration fraction 0.25, Table S1); PLS with A components fitted on the
    remaining units; standard finite-sample quantile of the absolute calibration residuals.
    ASSUMPTION: the interval is centred on the final model fitted on all calibration spectra ('the same final model',
    Section 3.3)."""
    rng = np.random.default_rng(seed)
    u = np.unique(units)
    cal_units = rng.choice(u, int(round(frac * len(u))), replace=False)
    cal = np.isin(units, cal_units)
    m = TruncPLS(X[~cal], y[~cal], A)
    r = np.sort(np.abs(y[cal] - m.predict(X[cal], A)))
    k = int(np.ceil((len(r) + 1) * (1 - alpha)))
    return float(r[min(k, len(r)) - 1])


def astm_group_interval(gc, X, alpha=0.1):
    """ASTM-G baseline: yhat -/+ t * RMSECV_group * sqrt(1 + h), with the RMSE of the out-of-group residuals of a fitted
    GroupConformal and the leverage h of the final model. ASSUMPTIONS: unweighted RMSE over all calibration spectra;
    t quantile with n - A - 1 degrees of freedom."""
    rmse = float(np.sqrt(np.mean(gc.resid ** 2)))
    tq = stats.t.ppf(1 - alpha / 2, gc.final.n_train - gc.A - 1)
    yhat = gc.final.predict(X, gc.A)
    half = tq * rmse * np.sqrt(1 + gc.final.leverage(X, gc.A))
    return yhat, yhat - half, yhat + half


def group_metrics(y, lo, hi, groups, alpha=0.1):
    """Group-averaged coverage, interval score and width (mean over held-out groups of the per-spectrum means)."""
    y, lo, hi, groups = map(np.asarray, (y, lo, hi, groups))
    out = {'coverage': [], 'interval_score': [], 'width': []}
    for g in np.unique(groups):
        m = groups == g
        out['coverage'].append(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
        out['interval_score'].append(np.mean(interval_score(lo[m], hi[m], y[m], alpha)))
        out['width'].append(np.mean(hi[m] - lo[m]))
    return {k: float(np.mean(v)) for k, v in out.items()} | {'min_group_coverage': float(np.min(out['coverage']))}

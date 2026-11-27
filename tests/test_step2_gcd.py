"""Step 2b (group-conformal baselines): the evaluation pipeline must reproduce Step 2 exactly when it is given the
Step 2 centre (PLS on Cal) and the Step 2 constant half-width, so rows of the two steps are paired on the same draws."""
import numpy as np
import pandas as pd
import pytest

from src.data import TARGET, fruit_table, set_masks, wavelength_columns, wavelengths
from src.models_pls import fit_pls, predict
from src.preprocess import preprocess
from src.step2_decision import KS, N_REP, SEED, TEST_SETS, THRESHOLDS, evaluate
from src.step2_gcd_baseline import apply_correction, draw_reference_fruit, metric_rows, step2_draws
from src.uncertainty import repeated_bias_correct


@pytest.fixture(scope='module')
def step2_fruit(df):
    X, _ = preprocess(df[wavelength_columns(df)].to_numpy(), wavelengths(df))
    cal = set_masks(df)['Cal']
    pred = predict(fit_pls(X[cal], df[TARGET].to_numpy()[cal], 11), X)
    return fruit_table(df, pred)


def test_draws_reproduce_repeated_bias_correct(step2_fruit):
    sub = step2_fruit[step2_fruit['Set'] == 'Val Ext 2']
    ours = apply_correction(sub, draw_reference_fruit(sub, 5, 4, np.random.default_rng(3)))
    ref = repeated_bias_correct(sub, 5, 4, np.random.default_rng(3))
    assert list(ours.index) == list(ref.index)
    np.testing.assert_allclose(ours['p'], ref['p'])
    np.testing.assert_array_equal(ours['rep'], ref['rep'])


def test_pipeline_reproduces_step2_rows(df, step2_fruit):
    res = evaluate(df)
    q = res['half_width']
    draws = step2_draws(df)
    n_checked = 0
    for s in TEST_SETS:
        sub = step2_fruit[step2_fruit['Set'] == s].assign(half=q)
        ours = {('none', r['T']): r for r in metric_rows(sub, THRESHOLDS, repeated=False)}
        for k in KS:
            ours |= {(f'k={k}', r['T']): r for r in metric_rows(apply_correction(sub, draws[(s, k)]), THRESHOLDS, True)}
        for r in (r for r in res['rows'] if r['set'] == s):
            o = ours[(r['correction'], r['T'])]
            for key in (k for k in r if k not in ('set', 'correction', 'T', 'half_width', 'n_rep')):  # incl. *_sd
                assert o[key] == pytest.approx(r[key], abs=1e-12, nan_ok=True), (s, r['correction'], r['T'], key)
                n_checked += 1
    assert n_checked == 2 * len(THRESHOLDS) * (10 + len(KS) * 14)   # 10 metrics per row, plus 4 SDs when repeated


def test_common_point_model_is_the_step1_model(df):
    """point_model 'A=11': Tuning-conformal centre and GC-D final model both equal the Step 1 PLS (11 LV, Cal + Tuning),
    and the Tuning half-width equals the Step 2 one."""
    from src.group_conformal import GroupConformal, TruncPLS
    from src.step2_gcd_baseline import TRAIN_SETS, tuning_conformal_halfwidth
    X, _ = preprocess(df[wavelength_columns(df)].to_numpy(), wavelengths(df))
    y = df[TARGET].to_numpy()
    train = df['Set'].isin(TRAIN_SETS).to_numpy()
    test = ~train
    step1 = predict(fit_pls(X[train], y[train], 11), X[test])
    np.testing.assert_allclose(TruncPLS(X[train], y[train], 11).predict(X[test], 11), step1, rtol=1e-9, atol=1e-9)
    gc = GroupConformal(alpha=0.1, level='unit', n_components=11).fit(X[train], y[train], df['Season'].to_numpy()[train],
                                                                     df['fruit'].to_numpy()[train])
    assert gc.A == 11
    np.testing.assert_allclose(gc.final.predict(X[test], gc.A), step1, rtol=1e-9, atol=1e-9)
    assert tuning_conformal_halfwidth(df, X, y) == pytest.approx(evaluate(df)['half_width'], rel=1e-12)

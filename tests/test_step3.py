"""Step 3 code that runs without TensorFlow: pretreatments, run configurations and the analysis of saved predictions.
The analysis is exercised with PLS predictions standing in for network outputs."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter

from src.data import DATA_PATH, TARGET, set_masks, wavelength_columns, wavelengths
from src.models_pls import fit_pls, predict, regression_metrics
from src.preprocess import augment, preprocess, snv
from src.step2_decision import evaluate
from src.step3_analysis import (BENCHMARKS, benchmark_comparison, decision_table, load_runs, prediction_interval,
                                regression_table, summarise_regression)
from src.step3_train_nn import EXTRA_SEEDS, FEATURE_VERSION, RUNS, SEEDS, features, plan, run_paths, run_status, supersede


def spectra(df, n=None):
    return df[wavelength_columns(df)].to_numpy()[:n], wavelengths(df)


def test_augment_after_trim_blocks(df):
    raw, wl = spectra(df, 20)
    keep = (wl >= 684) & (wl <= 990)
    S = snv(raw, (wl >= 402) & (wl <= 1098), ddof=1)  # SNV statistics over 402-1098 nm, as in Walsh's files
    A = augment(raw, wl, trim_first=False)
    assert A.shape == (20, 618)
    blocks = np.split(A, 6, axis=1)
    np.testing.assert_allclose(blocks[0], raw[:, keep])
    np.testing.assert_allclose(blocks[1], S[:, keep])
    np.testing.assert_allclose(blocks[3], savgol_filter(raw, 13, 2, deriv=2, axis=1)[:, keep])
    np.testing.assert_allclose(blocks[5], savgol_filter(S, 13, 2, deriv=2, axis=1)[:, keep])


def test_augment_trim_first_blocks(df):
    raw, wl = spectra(df, 20)
    t = raw[:, (wl >= 684) & (wl <= 990)]
    blocks = np.split(augment(raw, wl, trim_first=True), 6, axis=1)
    np.testing.assert_allclose(blocks[1].mean(axis=1), 0, atol=1e-12)  # SNV over the 103 trimmed variables
    np.testing.assert_allclose(blocks[1].std(axis=1, ddof=1), 1)
    np.testing.assert_allclose(blocks[2], savgol_filter(t, 13, 2, deriv=1, axis=1))
    np.testing.assert_allclose(blocks[4], savgol_filter(snv(t, ddof=1), 13, 2, deriv=1, axis=1))


@pytest.mark.parametrize('name, trim_first', [('mishra_pretreatment', True), ('mishra_pretreatment_adjusted', False)])
def test_augment_reproduces_walsh_input_files(df, name, trim_first):
    """Walsh et al. 2024 published their pretreated season-5 inputs (Mendeley Data doi:10.17632/mc9jshwb38.1,
    input/season_5/mishra&passos_2021_<name>.csv): 'TA' and 'AT' must reproduce them block by block."""
    path = DATA_PATH.parent / f'Walsh2024_season5_{name}.csv'
    if not path.exists():
        pytest.skip(f'{path.name} not found (optional, see README)')
    w = pd.read_csv(path)
    v2 = df[df['Set'] == 'Val Ext 2']
    np.testing.assert_array_equal(w['dm'].to_numpy(), v2[TARGET].to_numpy())  # same spectra, same order
    theirs = w[[c for c in w.columns if c[:1].isdigit()]].to_numpy()
    ours = augment(v2[wavelength_columns(df)].to_numpy(), wavelengths(df), trim_first=trim_first)
    for i in range(6):
        b = slice(103 * i, 103 * (i + 1))
        np.testing.assert_allclose(ours[:, b], theirs[:, b], rtol=0, atol=1e-12 * np.abs(theirs[:, b]).max())


def test_run_configurations(df):
    raw, wl = spectra(df, 5)
    n_features = {'S': 103, 'TA': 618, 'AT': 618}
    for name, cfg in RUNS.items():
        assert cfg['model'] in ('ann', 'cnn') and cfg['train'] in ('cal', 'cal+tuning')
        assert features(raw, wl, cfg['pretreatment']).shape == (5, n_features[cfg['pretreatment']]), name
    # the decision-layer run must not touch the Tuning set, which calibrates the conformal interval
    assert RUNS['cnn_AT_calonly'] == dict(model='cnn', pretreatment='AT', train='cal', monitor='loss', scaler_fit='cal')


@pytest.fixture(scope='module')
def pls_cal_pred(df):
    X, _ = preprocess(*spectra(df))
    cal = set_masks(df)['Cal']
    y = df[TARGET].to_numpy()
    return predict(fit_pls(X[cal], y[cal], 11), X)


def test_analysis_reads_saved_predictions(df, pls_cal_pred, tmp_path):
    (tmp_path / 'preds').mkdir()
    for seed in (0, 1):
        pd.DataFrame({'row': np.arange(len(df)), 'pred': pls_cal_pred}).to_csv(
            tmp_path / 'preds' / f'pls_seed{seed}.csv', index=False, float_format='%.5f')
    runs = load_runs(tmp_path, len(df))
    assert sorted(runs['pls']) == [0, 1]
    summary = summarise_regression(regression_table(df, runs)).set_index('set')
    m = set_masks(df)['Val Ext']
    expected = regression_metrics(df[TARGET].to_numpy()[m], pls_cal_pred[m])['rmsep']
    assert summary.loc['Val Ext', 'rmsep_mean'] == pytest.approx(expected, abs=1e-4)  # CSV keeps 5 decimals
    assert summary.loc['Val Ext', 'rmsep_sd'] == 0


def test_decision_table_matches_step2(df, pls_cal_pred):
    ours = decision_table(df, {0: pls_cal_pred}).drop(columns='seed')
    step2 = pd.DataFrame(evaluate(df)['rows'])
    pd.testing.assert_frame_equal(ours, step2)


def test_stale_runs_are_superseded(tmp_path):
    pred, meta = run_paths(tmp_path, 'cnn_AT_cal', 0)
    assert run_status(tmp_path, 'cnn_AT_cal', 0, 2) == 'missing'
    pred.parent.mkdir(parents=True)
    meta.parent.mkdir(parents=True)
    pred.write_text('row,pred\n')
    meta.write_text('{"run": "cnn_AT_cal"}')  # first Colab run: no feature_version -> version 1
    assert run_status(tmp_path, 'cnn_AT_cal', 0, 1) == 'done'
    assert run_status(tmp_path, 'cnn_AT_cal', 0, 2) == 'stale'
    assert supersede(tmp_path, 'cnn_AT_cal', 0) == 1
    assert not pred.exists() and not meta.exists()
    assert (tmp_path / 'superseded' / 'v1' / 'preds' / pred.name).exists()
    assert (tmp_path / 'superseded' / 'v1' / 'runs' / meta.name).exists()
    assert run_status(tmp_path, 'cnn_AT_cal', 0, 2) == 'missing'
    # superseded predictions are not picked up by the analysis
    assert load_runs(tmp_path, 1) == {}


def test_feature_versions_and_plan():
    assert set(FEATURE_VERSION) == {cfg['pretreatment'] for cfg in RUNS.values()}
    assert FEATURE_VERSION['AT'] == 2  # SNV over 402-1098 nm since the second Colab run
    full = plan(list(RUNS))
    assert full[:len(RUNS)] == [(r, SEEDS[0]) for r in RUNS]  # seeds form the outer loop
    assert len(full) == len(RUNS) * len(SEEDS) + sum(len(s) for s in EXTRA_SEEDS.values())
    assert [s for r, s in full if r == 'cnn_TA_cal'] == list(range(20))
    assert plan(['cnn_TA_cal'], seeds=[7]) == [('cnn_TA_cal', 7)]  # explicit seeds: no extra seeds
    assert plan(list(RUNS), smoke=True) == [(r, 0) for r in RUNS]


def test_prediction_interval():
    lo, hi = prediction_interval([1.0, 2.0, 3.0])
    h = 4.302652729911275 * 1.0 * np.sqrt(1 + 1 / 3)  # t(0.975, 2) * SD * sqrt(1 + 1/n)
    assert (lo, hi) == pytest.approx((2 - h, 2 + h))
    assert np.isnan(prediction_interval([1.0])[0])


def test_benchmark_comparison_criteria():
    b = BENCHMARKS['cnn_TA_cal']['rmsep']
    reg = pd.DataFrame({'run': 'cnn_TA_cal', 'set': 'Val Ext', 'seed': range(5),
                        'rmsep': b + 0.08 + np.array([-0.06, -0.03, 0.0, 0.03, 0.06]), 'bias': 0.0})
    c = benchmark_comparison(reg)[0]
    assert c['rmsep_diff'] == pytest.approx(0.08) and not c['equivalent']  # mean too far from Walsh's single run
    assert c['within_pi']  # but Walsh's run is a plausible single draw

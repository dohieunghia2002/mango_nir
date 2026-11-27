"""Step 4b without TensorFlow: analyse_set with a small nonlinear numpy 'network' whose gradient is known; saving of the
models; comparison with Step 3; the summary.json entry."""
import json

import numpy as np
import pandas as pd
import pytest

import src.step4b_explain_cnn as s4b
from src.explain import band_slices, block_bands, unit_means
from src.step4b_explain_cnn import analyse_set


def test_analyse_set_with_a_numpy_network():
    rng = np.random.default_rng(0)
    p = 618
    W = rng.normal(scale=0.05, size=(p, 8)).astype(np.float32)
    v = rng.normal(size=8).astype(np.float32)
    predict = lambda X: 15.0 + np.tanh(np.asarray(X, np.float32) @ W) @ v           # nonlinear in x
    grad = lambda X: ((1 - np.tanh(np.asarray(X, np.float32) @ W) ** 2) * v) @ W.T   # its exact gradient
    X = rng.normal(size=(120, p)).astype(np.float32)
    units = np.repeat(np.arange(60), 2)
    pops = units // 20
    bands = block_bands(band_slices(103), 6, 103)
    labels = [f'b{j}' for j in range(10)]
    baseline = np.zeros(p, dtype=np.float32)

    def decider(Xr, ur):
        ids, c = unit_means(predict(Xr), ur)
        return ids, c, np.full(len(ids), 0.5)

    comp, faith, stab, rec = analyse_set(predict, grad, decider, X, units, pops, bands, labels, baseline,
                                         float(predict(baseline[None])[0]), ig_steps=64, n_draws=10)
    # IG completeness with an exact gradient: small in %FW and relative to the change from the baseline
    assert comp['abs_error']['max'] < 1e-3 and comp['rel_error']['median'] < 1e-3
    assert comp['abs_error']['median'] <= comp['abs_error']['p95'] <= comp['abs_error']['max']
    assert {r['method'] for r in faith} == {'IG', 'GxI'}
    assert len(faith) == 2 * 10 * 2                               # methods x m x (top, random)
    assert len(rec) == 2 * 60 and set(rec['decision']) <= {'ready', 'not_ready', 'defer'}
    # deleting all 10 bands puts every spectrum at the baseline: top and random give the same change
    for method in ('IG', 'GxI'):
        t = [r for r in faith if r['method'] == method and r['m'] == 10]
        assert t[0]['abs_change'] == pytest.approx(t[1]['abs_change'])
    # IG band attributions of a fruit sum (over bands) to the change of its prediction from the baseline
    ig = rec[rec['method'] == 'IG']
    band_total = ig[[f'band_{j + 1}' for j in range(10)]].sum(axis=1).to_numpy()
    np.testing.assert_allclose(band_total, ig['p'].to_numpy() - float(predict(baseline[None])[0]), atol=5e-3)


class FakeModel:
    """Stands for a Keras model: save() writes the file in two steps and can fail in between (a Colab disconnect)."""

    def __init__(self, fail=False):
        self.fail = fail

    def save(self, path):
        assert path.suffix == '.keras'                  # Keras 3 saves only to names ending in '.keras'
        path.write_bytes(b'first part')
        if self.fail:
            raise RuntimeError('disconnected')
        path.write_bytes(b'complete model')


def test_model_appears_under_its_final_name_only_when_saved_with_its_metadata(tmp_path):
    model_dir = tmp_path / 'models'
    path, meta_path = s4b.model_paths(model_dir, 0)
    with pytest.raises(RuntimeError):
        s4b.save_model(FakeModel(fail=True), {'epochs': 412}, model_dir, 0)
    assert not path.exists() and not meta_path.exists()          # the next run trains this seed again
    assert s4b.remove_partial_files(tmp_path) == [s4b.partial_path(path)]
    assert list(model_dir.iterdir()) == []
    s4b.save_model(FakeModel(), {'epochs': 412}, model_dir, 0)
    assert path.read_bytes() == b'complete model'
    assert json.loads(meta_path.read_text(encoding='utf-8')) == {'epochs': 412}
    assert s4b.remove_partial_files(tmp_path) == [] and s4b.remove_partial_files(tmp_path / 'missing') == []


def test_compare_with_step3_uses_the_stored_predictions(df):
    stored = pd.read_csv(s4b.run_paths(s4b.RESULTS / 'step3', s4b.RUN, 0)[0]).sort_values('row')['pred'].to_numpy()
    close = s4b.compare_with_step3(df, stored + 0.5 * s4b.STEP3_TOLERANCE, 0)
    assert close['match'] and close['rmsep_new_Val Ext'] == pytest.approx(close['rmsep_step3_Val Ext'], abs=1e-4)
    assert not s4b.compare_with_step3(df, stored + 2 * s4b.STEP3_TOLERANCE, 0)['match']
    assert s4b.compare_with_step3(df, stored, 99) == {'stored_predictions': False, 'match': False}


def fake_detail(smoke=False, seeds=s4b.SEEDS):
    check = {'stored_predictions': True, 'max_abs_diff': 3e-6, 'match': True,
             **{f'rmsep_{w}_{s}': 0.9 for w in ('new', 'step3') for s in ('Val Ext', 'Val Ext 2')}}
    env = {'python': '3.13', 'tensorflow': '2.21.0', 'keras': '3.13.2', 'gpu': ['/physical_device:GPU:0'],
           'gpu_names': ['Tesla T4'], 'git_commit': 'abcdef123456'}
    return {'config': {'smoke': smoke, 'seeds': list(seeds)}, 'environment': env,
            'per_seed': [{'seed': s, 'loaded': False, 'step3_check': check, 'train_meta': {'environment': env}}
                         for s in seeds],
            'aggregate_faithfulness': [], 'aggregate_stability': []}


def write_results(folder, detail):
    (folder / 'step4b.json').write_text(json.dumps(detail), encoding='utf-8')
    (folder / 'summary.json').write_text('{"step4": {}}', encoding='utf-8')


@pytest.mark.parametrize('smoke, seeds', [(True, (0,)), (False, (0,))])
def test_summary_only_refuses_smoke_or_incomplete_results(tmp_path, monkeypatch, smoke, seeds):
    monkeypatch.setattr(s4b, 'RESULTS', tmp_path)
    write_results(tmp_path, fake_detail(smoke, seeds))
    with pytest.raises(SystemExit):
        s4b.write_summary()
    assert json.loads((tmp_path / 'summary.json').read_text(encoding='utf-8')) == {'step4': {}}


def test_summary_only_prints_the_step3_check_and_writes_the_entry(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(s4b, 'RESULTS', tmp_path)
    write_results(tmp_path, fake_detail())
    s4b.write_summary()
    out = capsys.readouterr().out
    assert out.count('vs Step 3: max |difference| 3.0e-06 %FW (same)') == len(s4b.SEEDS) and 'WARNING' not in out
    assert 'GPU Tesla T4' in out
    summary = json.loads((tmp_path / 'summary.json').read_text(encoding='utf-8'))
    assert set(summary) == {'step4', 'step4b'}
    assert set(s4b.SUMMARY_KEYS) <= set(summary['step4b']) and summary['step4b']['environment']['keras'] == '3.13.2'

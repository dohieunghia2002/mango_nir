"""Step 3 analysis - RMSEP of the ANN / 1D-CNN runs against Walsh et al. 2024, then the Step 2 decision layer with the CNN.

Reads the predictions written by src.step3_train_nn. Light computation: runs locally or at the end of the Colab notebook.
  python -m src.step3_analysis [--runs-dir results/step3]
Writes results/step3_regression.csv (per run, seed and set), results/step3_decision_metrics.csv (per seed),
results/step3.json and the 'step3' entry of results/summary.json. Predictions made with older features are left out.
If a planned run has no current predictions, the results are still written (marked 'complete': false) and the script
exits with code 4, so the notebook does not package them as final.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data import ROOT, TARGET, load_data, set_masks
from src.models_pls import regression_metrics
from src.step2_decision import evaluate_predictions
from src.step3_train_nn import FEATURE_VERSION, RUNS, plan, run_status

RESULTS = ROOT / 'results'
DECISION_RUN = 'cnn_AT_calonly'
EVAL_SETS = ['Tuning', 'Val Ext', 'Val Ext 2']
EQUIVALENT_TOL = 0.05  # about twice the seed-to-seed SD of RMSEP in Walsh 2024, Table 2 (0.02-0.03 %FW)
PI_LEVEL = 0.95
EXIT_INCOMPLETE = 4

# Walsh 2024 models trained like ours: Set partition (KS population), file order, no extra outlier removal,
# standardised inputs. Values from their Appendix A (Mendeley Data doi:10.17632/mc9jshwb38.1, appendix_a_model_results.xlsx),
# where each model is a single training run (seed 42). In raw_results.csv, A002 and C006 appear twice under two
# set_split labels with the same training data; their results agree to 1e-15, so Walsh's training was deterministic.
# Walsh reports bias as mean(true - pred); converted here to our mean(pred - true).
BENCHMARKS = {
    'ann_S_cal': {'source': 'Walsh 2024, Appendix A, A026 (also Table 5)', 'rmsep': 0.958, 'bias': 0.013},
    'ann_S_all': {'source': 'Walsh 2024, Appendix A, A002 (Tables 1, 3; Anderson 2020 ANN: 0.89)', 'rmsep': 0.893, 'bias': -0.080},
    'cnn_TA_cal': {'source': 'Walsh 2024, Appendix A, C028 (also Table 5)', 'rmsep': 0.824, 'bias': 0.006},
    'cnn_AT_cal': {'source': 'Walsh 2024, Appendix A, C030', 'rmsep': 0.814, 'bias': 0.067},
    'cnn_AT_all': {'source': 'Walsh 2024, Appendix A, C006', 'rmsep': 0.827, 'bias': -0.027},
    'cnn_TA_cal_k2': {'source': 'Walsh 2024, Appendix A, C028 (our run with Keras 2)', 'rmsep': 0.824, 'bias': 0.006},
}
# Same configuration trained with Keras 2 (tf_keras) and Keras 3: tests whether the Keras version explains the gap
# between our CNN runs and Walsh's (TF 2.11, Keras 2).
KERAS_PAIRS = {'cnn_TA_cal_k2': 'cnn_TA_cal'}
REFERENCE_NOTE = ('Not comparable, reference only: Walsh 2024 best models used the extra outlier removal of Mishra & Passos '
                  '2021 and/or their random sample partition. CNN C134: season 4 RMSEP 0.77, season 5 RMSEP 1.18 '
                  '(bias -0.16 in our convention); ANN A010: 0.88 and 2.62; PLS P076: 1.00 and 1.87.')
DECISION_METRICS = ['half_width', 'truly_ready', 'accuracy_no_defer', 'false_ready_no_defer', 'interval_coverage',
                    'decided', 'accuracy_decided', 'false_ready_decided', 'missed_ready_decided']


def records(table):
    """DataFrame rows as dicts, with NaN (e.g. the SD of a single seed) as None so the JSON stays valid."""
    return table.astype(object).where(table.notna(), None).to_dict('records')


def load_runs(runs_dir, n_rows, check_version=True):
    """({run: {seed: per-spectrum predictions}}, excluded) from <runs_dir>/preds/<run>_seed<k>.csv.
    Predictions of a configuration in RUNS made with an older feature version (step3_train_nn.FEATURE_VERSION, read
    from runs/<run>_seed<k>.json) are left out and listed in `excluded`, so a partly re-run set of runs never mixes
    old and new features. check_version=False reads every file (used for the superseded runs)."""
    runs_dir = Path(runs_dir)
    runs, excluded = {}, []
    for f in sorted((runs_dir / 'preds').glob('*_seed*.csv')):
        run, seed = f.stem.rsplit('_seed', 1)
        seed = int(seed)
        if (check_version and run in RUNS
                and run_status(runs_dir, run, seed, FEATURE_VERSION[RUNS[run]['pretreatment']]) == 'stale'):
            excluded.append(f.stem)
            continue
        p = pd.read_csv(f)
        if len(p) != n_rows:
            raise ValueError(f'{f}: {len(p)} rows, expected {n_rows}')
        pred = np.full(n_rows, np.nan)
        pred[p['row'].to_numpy()] = p['pred'].to_numpy()
        runs.setdefault(run, {})[seed] = pred
    return runs, excluded


def missing_runs(runs):
    """Runs planned by step3_train_nn (all configurations, SEEDS and EXTRA_SEEDS) without current predictions."""
    return [f'{r}_seed{s}' for r, s in plan(list(RUNS)) if s not in runs.get(r, {})]


def finite_or_none(x):
    return float(x) if np.isfinite(x) else None


def run_devices(runs_dir):
    """{(run, seed): 'gpu' or 'cpu'} from the environment recorded in <runs_dir>/runs/<run>_seed<k>.json."""
    out = {}
    for f in (Path(runs_dir) / 'runs').glob('*_seed*.json'):
        m = json.loads(f.read_text(encoding='utf-8'))
        out[(m['run'], int(m['seed']))] = 'gpu' if m['environment'].get('gpu') else 'cpu'
    return out


def device_comparison(reg, devices):
    """For configurations trained partly on GPU and partly on CPU (a Colab session without GPU): RMSEP per device
    and Welch's t-test, per set. Needs at least 2 seeds on each device."""
    d = reg.assign(device=[devices.get((r, s)) for r, s in zip(reg['run'], reg['seed'])])
    out = []
    for (run, s), g in d.groupby(['run', 'set'], sort=False):
        cpu, gpu = g.loc[g['device'] == 'cpu', 'rmsep'], g.loc[g['device'] == 'gpu', 'rmsep']
        if len(cpu) >= 2 and len(gpu) >= 2:
            out.append({'run': run, 'set': s, 'n_cpu': int(len(cpu)), 'n_gpu': int(len(gpu)),
                        'rmsep_mean_cpu': float(cpu.mean()), 'rmsep_mean_gpu': float(gpu.mean()),
                        'welch_p': float(stats.ttest_ind(cpu, gpu, equal_var=False).pvalue)})
    return out


def regression_table(df, runs):
    y = df[TARGET].to_numpy()
    masks = set_masks(df)
    return pd.DataFrame([{'run': run, 'seed': seed, 'set': s, **regression_metrics(y[masks[s]], pred[masks[s]])}
                         for run, seeds in runs.items() for seed, pred in sorted(seeds.items()) for s in EVAL_SETS])


def summarise_regression(reg):
    g = reg.groupby(['run', 'set'])
    out = g.agg(n_seeds=('seed', 'nunique'), rmsep_mean=('rmsep', 'mean'), rmsep_sd=('rmsep', 'std'),
                rmsep_min=('rmsep', 'min'), rmsep_max=('rmsep', 'max'), bias_mean=('bias', 'mean'), sep_mean=('sep', 'mean'))
    return out.reset_index()


def prediction_interval(values, level=PI_LEVEL):
    """Interval expected to contain one new run with probability `level` (t distribution, n - 1 df):
    mean +- t * SD * sqrt(1 + 1/n). Fits the comparison with Walsh's single runs."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return float('nan'), float('nan')
    h = stats.t.ppf((1 + level) / 2, v.size - 1) * v.std(ddof=1) * np.sqrt(1 + 1 / v.size)
    return float(v.mean() - h), float(v.mean() + h)


def benchmark_comparison(reg):
    """Season 4 RMSEP against Walsh's single runs with two criteria: |mean over seeds - Walsh| <= EQUIVALENT_TOL
    ('equivalent', set before the runs), and Walsh's value inside the PI_LEVEL prediction interval of one run
    ('within_pi', added after Walsh's values turned out to be single runs)."""
    out = []
    for run, b in BENCHMARKS.items():
        v = reg[(reg['run'] == run) & (reg['set'] == 'Val Ext')]
        if v.empty:
            continue
        diff = float(v['rmsep'].mean() - b['rmsep'])
        lo, hi = prediction_interval(v['rmsep'])
        has_pi = bool(np.isfinite(lo))  # needs at least 2 seeds
        out.append({'run': run, **b, 'rmsep_mean': float(v['rmsep'].mean()), 'rmsep_sd': finite_or_none(v['rmsep'].std()),
                    'bias_mean': float(v['bias'].mean()), 'n_seeds': int(v['seed'].nunique()),
                    'n_seeds_better_than_benchmark': int((v['rmsep'] < b['rmsep']).sum()), 'rmsep_diff': diff,
                    'equivalent': bool(abs(diff) <= EQUIVALENT_TOL), 'pi_low': finite_or_none(lo),
                    'pi_high': finite_or_none(hi), 'within_pi': bool(lo <= b['rmsep'] <= hi) if has_pi else None})
    return out


def keras_comparison(reg):
    """RMSEP of each KERAS_PAIRS configuration with Keras 2 vs Keras 3, per set: means, SDs and Welch's t-test.
    Seeds are not paired: the two Keras versions draw different random numbers from the same seed."""
    out = []
    for k2, k3 in KERAS_PAIRS.items():
        for s in EVAL_SETS:
            a = reg[(reg['run'] == k2) & (reg['set'] == s)]['rmsep'].to_numpy()
            b = reg[(reg['run'] == k3) & (reg['set'] == s)]['rmsep'].to_numpy()
            if a.size < 2 or b.size < 2:
                continue
            out.append({'run_keras2': k2, 'run_keras3': k3, 'set': s, 'n_keras2': int(a.size), 'n_keras3': int(b.size),
                        'rmsep_mean_keras2': float(a.mean()), 'rmsep_sd_keras2': float(a.std(ddof=1)),
                        'rmsep_mean_keras3': float(b.mean()), 'rmsep_sd_keras3': float(b.std(ddof=1)),
                        'diff_keras2_minus_keras3': float(a.mean() - b.mean()),
                        'welch_p': float(stats.ttest_ind(a, b, equal_var=False).pvalue)})
    return out


def decision_table(df, seeds):
    rows = []
    for seed, pred in sorted(seeds.items()):
        res = evaluate_predictions(df, pred)
        rows += [{'seed': seed, **r} for r in res['rows']]
    return pd.DataFrame(rows)


def summarise_decision(dec):
    g = dec.groupby(['set', 'correction', 'T'], sort=False)[DECISION_METRICS]
    mean, sd = g.mean().add_suffix('_mean'), g.std().add_suffix('_sd')
    return pd.concat([mean, sd], axis=1).reset_index().assign(n_seeds=dec['seed'].nunique())


def superseded_analysis(df, runs_dir):
    """Runs replaced after a change of features (<runs_dir>/superseded/v<k>/), kept as a sensitivity analysis:
    {'v<k>': {'runs', 'regression_summary', 'decision_summary'}}; the decision layer uses DECISION_RUN if present."""
    out = {}
    for d in sorted((Path(runs_dir) / 'superseded').glob('v*')):
        runs, _ = load_runs(d, len(df), check_version=False)
        if not runs:
            continue
        decision = (records(summarise_decision(decision_table(df, runs[DECISION_RUN])))
                    if DECISION_RUN in runs else None)
        out[d.name] = {'runs': {r: sorted(s) for r, s in runs.items()},
                       'regression_summary': records(summarise_regression(regression_table(df, runs))),
                       'decision_summary': decision}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', type=Path, default=RESULTS / 'step3')
    args = ap.parse_args()

    df = load_data()
    runs, excluded = load_runs(args.runs_dir, len(df))
    if not runs:
        raise SystemExit(f'no current predictions found in {args.runs_dir / "preds"}')
    missing = missing_runs(runs)
    if excluded:
        print(f'WARNING: {len(excluded)} predictions made with older features are left out: {", ".join(excluded)}')
    if missing:
        print(f'WARNING: {len(missing)} planned runs have no current predictions: {", ".join(missing)}')
    reg = regression_table(df, runs)
    summary = summarise_regression(reg)

    print('RMSEP and bias (pred - true) in %FW, mean (SD) over seeds:')
    for _, r in summary.iterrows():
        print(f'  {r["run"]:<15} {r["set"]:<10} seeds {r["n_seeds"]} | RMSEP {r["rmsep_mean"]:.3f} ({r["rmsep_sd"]:.3f}), '
              f'range {r["rmsep_min"]:.3f}-{r["rmsep_max"]:.3f} | bias {r["bias_mean"]:+.3f}')
    comparison = benchmark_comparison(reg)
    print(f'\nSeason 4 vs comparable Walsh 2024 models (single runs): mean within +-{EQUIVALENT_TOL}, '
          f'and Walsh inside the {PI_LEVEL:.0%} prediction interval of one of our runs:')
    for c in comparison:
        pi = ('PI needs 2 seeds' if c['within_pi'] is None else
              f'PI [{c["pi_low"]:.3f}, {c["pi_high"]:.3f}] -> {"TRONG" if c["within_pi"] else "NGOÀI"}')
        print(f'  {c["run"]:<15} {c["rmsep_mean"]:.3f} ({c["n_seeds"]} seeds) vs {c["rmsep"]:.3f} ({c["source"]}), '
              f'diff {c["rmsep_diff"]:+.3f} -> {"TƯƠNG ĐƯƠNG" if c["equivalent"] else "KHÁC"} | {pi} | '
              f'seeds better than Walsh: {c["n_seeds_better_than_benchmark"]}/{c["n_seeds"]}')
    print(f'  {REFERENCE_NOTE}')
    keras = keras_comparison(reg)
    if keras:
        print('\nKeras 2 (tf_keras) vs Keras 3, same configuration, RMSEP mean (SD), Welch t-test:')
        for k in keras:
            print(f'  {k["run_keras2"]} vs {k["run_keras3"]:<11} {k["set"]:<10} '
                  f'{k["rmsep_mean_keras2"]:.3f} ({k["rmsep_sd_keras2"]:.3f}, n={k["n_keras2"]}) vs '
                  f'{k["rmsep_mean_keras3"]:.3f} ({k["rmsep_sd_keras3"]:.3f}, n={k["n_keras3"]}) | '
                  f'diff {k["diff_keras2_minus_keras3"]:+.3f}, p = {k["welch_p"]:.3f}')
    devices = device_comparison(reg, run_devices(args.runs_dir))
    if devices:
        print('\nConfigurations trained partly on CPU, RMSEP mean CPU vs GPU, Welch t-test:')
        for d in devices:
            print(f'  {d["run"]:<15} {d["set"]:<10} CPU {d["rmsep_mean_cpu"]:.3f} (n={d["n_cpu"]}) vs '
                  f'GPU {d["rmsep_mean_gpu"]:.3f} (n={d["n_gpu"]}) | p = {d["welch_p"]:.3f}')
    superseded = superseded_analysis(df, args.runs_dir)
    for version, s in superseded.items():
        print(f'\nSuperseded runs {version} (sensitivity analysis), RMSEP mean (SD):')
        for r in s['regression_summary']:
            if r['set'] != 'Tuning':
                print(f'  {r["run"]:<15} {r["set"]:<10} seeds {r["n_seeds"]} | {r["rmsep_mean"]:.3f} ({r["rmsep_sd"]:.3f})')

    decision = None
    if DECISION_RUN in runs:
        dec = decision_table(df, runs[DECISION_RUN])
        decision = summarise_decision(dec)
        cols = ['interval_coverage', 'decided', 'accuracy_decided', 'false_ready_decided']
        print(f'\nDecision layer with {DECISION_RUN}, T = 15, mean (SD) over {dec["seed"].nunique()} seeds, %:')
        for _, r in decision[decision['T'] == 15.0].iterrows():
            vals = ' | '.join(f'{c} {100 * r[c + "_mean"]:.1f} ({100 * r[c + "_sd"]:.1f})' for c in cols)
            print(f'  {r["set"]:<10} {r["correction"]:<5} q {r["half_width_mean"]:.3f} | {vals}')
        dec.to_csv(RESULTS / 'step3_decision_metrics.csv', index=False)
    else:  # do not leave the decision metrics of an earlier analysis next to the new results
        (RESULTS / 'step3_decision_metrics.csv').unlink(missing_ok=True)

    reg.to_csv(RESULTS / 'step3_regression.csv', index=False)
    detail = {'complete': not missing, 'missing_runs': missing, 'excluded_older_features': excluded,
              'runs': {run: sorted(seeds) for run, seeds in runs.items()}, 'equivalent_tolerance': EQUIVALENT_TOL,
              'prediction_interval_level': PI_LEVEL,
              'regression_summary': records(summary), 'benchmark_comparison': comparison, 'keras_comparison': keras,
              'device_comparison': devices, 'reference_note': REFERENCE_NOTE, 'decision_run': DECISION_RUN,
              'decision_summary': None if decision is None else records(decision), 'superseded': superseded}
    # allow_nan=False: a NaN left anywhere fails here instead of writing invalid JSON
    (RESULTS / 'step3.json').write_text(json.dumps(detail, indent=2, allow_nan=False), encoding='utf-8')
    summary_path = RESULTS / 'summary.json'
    all_summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    all_summary['step3'] = {
        'description': 'ANN and 1D-CNN (Walsh 2024 settings) on the fixed Set partition; spectrum-level RMSEP over seeds; '
                       f'decision layer of Step 2 with {DECISION_RUN} (Tuning set unused in training)',
        **{k: detail[k] for k in ('complete', 'missing_runs', 'runs', 'regression_summary', 'benchmark_comparison',
                                  'keras_comparison', 'device_comparison', 'decision_summary', 'superseded')},
    }
    # the step3 entry was checked for NaN with step3.json above; other steps are written as they were read
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding='utf-8')
    print(f'\nResults written to {RESULTS}')
    if missing:
        print(f'INCOMPLETE: {len(missing)} planned runs are missing; these results are not final.')
        sys.exit(EXIT_INCOMPLETE)


if __name__ == '__main__':
    main()

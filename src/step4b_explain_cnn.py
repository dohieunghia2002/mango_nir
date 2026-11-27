"""Step 4b - explanations of the CNN harvest decisions and checks of their faithfulness. Runs on Colab (GPU).

Design fixed on 26/11/2026 before running (agreed with the user):
- model: 'cnn_AT_calonly' of Step 3 (trained on Cal, Tuning unused), seeds 0-4. Step 3 saved predictions, not weights,
  so the models are trained again with the same code and seeds and saved; their predictions are compared with the
  Step 3 predictions (results/step3/preds) to check that they are the same models;
- decision layer as in Step 3: fruit centre = mean prediction of the fruit's spectra, half-width = conformal quantile
  of the Tuning fruit residuals of the same model; T = 14, 15, 16 %FW (15 reported); no bias correction;
- attributions on the 618 standardised inputs: Integrated Gradients (IG, 64 steps, baseline = training mean, i.e. 0
  after standardisation; completeness error reported in %FW and relative) and gradient x input (GxI); band
  attribution = sum over the band's wavelengths in all 6 pretreatment blocks, with the 10 bands of Step 4a;
- faithfulness: deletion of the m top-ranked bands of each fruit (|band attribution|) to the baseline vs m random bands
  (100 draws, the same draws for both methods), m = 1..10, outcomes as in Step 4a. Not circular: the network is
  nonlinear, so a band's attribution is not the effect of deleting that band alone;
- stability across the scans of a fruit, as in Step 4a.

Safeguards for the Colab run (27/11/2026, agreed with the user):
- models are trained only in a session with a GPU, as in Step 3 (the smoke run may use the CPU), otherwise exit code 5;
  the environment (Python, TensorFlow, Keras, GPU, git commit) is saved with each model and in the results;
- once a seed's model is trained or loaded, its predictions are compared with Step 3; a difference above
  STEP3_TOLERANCE stops the script before the seed is analysed (exit code 4). The saved model is kept, and
  --accept-step3-mismatch continues anyway (recorded in the results);
- a model is saved under a temporary name, then its metadata is written, then the model is renamed: a model file under
  its final name is complete and has its metadata. Temporary files left by an interruption are removed at the start;
- no new seed starts after --time-budget-hours (exit code 3); a rerun loads the saved models and continues.

On Colab:  python -m src.step4b_explain_cnn --out <dir> [--smoke] [--accept-step3-mismatch]
Locally, after unzipping step4b_results.zip into the repo root:  python -m src.step4b_explain_cnn --summary-only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import ROOT, TARGET, fruit_table, load_data, set_masks, wavelength_columns, wavelengths
from src.decision import decide
from src.explain import (band_labels, band_slices, band_sums, block_bands, deletion_test, stability, unit_means)
from src.explain_nn import completeness_error, gradient_x_input, integrated_gradients
from src.models_pls import regression_metrics
from src.preprocess import PreprocessConfig
from src.step2_decision import ALPHA, SEED, TEST_SETS, THRESHOLDS
from src.step3_train_nn import (EXIT_TIME_BUDGET, FEATURE_VERSION, RUNS, TIME_BUDGET_HOURS, environment, features,
                                run_paths)
from src.uncertainty import conformal_quantile

RESULTS = ROOT / 'results'
RUN = 'cnn_AT_calonly'
SEEDS = (0, 1, 2, 3, 4)
IG_STEPS = 64
N_DRAWS = 100
METHODS = ('IG', 'GxI')
T_MAIN = 15.0
N_BLOCKS, BLOCK_SIZE = 6, 103
SUMMARY_KEYS = ('config', 'environment', 'per_seed', 'aggregate_faithfulness', 'aggregate_stability')
# Same model as Step 3: max |prediction difference| over all spectra, in %FW. Step 3 stored its predictions with 5
# decimals and predicted with model.predict; here the model is called directly, in larger batches.
STEP3_TOLERANCE = 1e-4
EXIT_STEP3_MISMATCH = 4
EXIT_NO_GPU = 5
PARTIAL = '.partial'  # temporary files are named <name>.partial<suffix>


def model_paths(model_dir, seed):
    return model_dir / f'{RUN}_seed{seed}.keras', model_dir / f'{RUN}_seed{seed}.json'


def partial_path(path):
    """Temporary name next to path that keeps its suffix (Keras saves only to names ending in '.keras')."""
    return path.with_name(path.stem + PARTIAL + path.suffix)


def write_text_via_partial(path, text):
    tmp = partial_path(path)
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def remove_partial_files(folder):
    """Remove the temporary files an interrupted run left in folder (recursively); returns their paths."""
    found = sorted(folder.rglob(f'*{PARTIAL}.*')) if folder.exists() else []
    for p in found:
        p.unlink()
    return found


def save_model(model, meta, model_dir, seed):
    """Model under a temporary name, then its metadata, then the model renamed to its final name."""
    path, meta_path = model_paths(model_dir, seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = partial_path(path)
    model.save(tmp)
    write_text_via_partial(meta_path, json.dumps(meta, indent=2))
    tmp.replace(path)


def train_or_load(seed, Xs, y, train_rows, model_dir, env, smoke):
    """The saved model of this seed, or cnn_AT_calonly trained as Step 3 does (training loss monitored, no validation
    data) and saved. Returns (model, metadata, loaded)."""
    import tensorflow as tf
    from src.models_nn import train
    path, meta_path = model_paths(model_dir, seed)
    if path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        return tf.keras.models.load_model(path, compile=False), meta, True
    model, meta = train('cnn', Xs[train_rows], y[train_rows], None, None, seed=seed, max_epochs=2 if smoke else None)
    meta.update({'run': RUN, 'seed': seed, 'n_train': int(train_rows.size), 'feature_version': FEATURE_VERSION['AT'],
                 'smoke': smoke, 'environment': env})
    save_model(model, meta, model_dir, seed)
    return model, meta, False


def session_environment():
    """The Step 3 environment record plus the GPU model names (e.g. 'Tesla T4')."""
    import tensorflow as tf
    env = environment()
    env['gpu_names'] = [tf.config.experimental.get_device_details(d).get('device_name', '?')
                        for d in tf.config.list_physical_devices('GPU')]
    return env


def env_text(env):
    if not env:
        return 'not recorded'
    gpu = ', '.join(env.get('gpu_names') or env.get('gpu') or ['none'])
    return (f"Python {env.get('python')}, TensorFlow {env.get('tensorflow')}, Keras {env.get('keras')}, GPU {gpu}, "
            f"commit {(env.get('git_commit') or '?')[:8]}")


def compare_with_step3(df, pred, seed):
    """Same model as Step 3? Max |difference| of the predictions over all spectra (match: at most STEP3_TOLERANCE)
    and RMSEP on the test sets, new vs stored."""
    path = run_paths(RESULTS / 'step3', RUN, seed)[0]
    if not path.exists():
        return {'stored_predictions': False, 'match': False}
    old = pd.read_csv(path).sort_values('row')['pred'].to_numpy()
    diff = float(np.max(np.abs(pred - old)))
    y = df[TARGET].to_numpy()
    masks = set_masks(df)
    out = {'stored_predictions': True, 'max_abs_diff': diff, 'match': diff <= STEP3_TOLERANCE}
    for s in TEST_SETS:
        out[f'rmsep_new_{s}'] = regression_metrics(y[masks[s]], pred[masks[s]])['rmsep']
        out[f'rmsep_step3_{s}'] = regression_metrics(y[masks[s]], old[masks[s]])['rmsep']
    return out


def check_line(seed, loaded, check, env):
    """Trained or loaded, the comparison with Step 3 and the environment the model was trained in."""
    if check['stored_predictions']:
        result = (f"max |difference| {check['max_abs_diff']:.1e} %FW ({'same' if check['match'] else 'DIFFERENT'}), "
                  'RMSEP new/Step 3 ' + ', '.join(f"{s} {check[f'rmsep_new_{s}']:.3f}/{check[f'rmsep_step3_{s}']:.3f}"
                                                 for s in TEST_SETS))
    else:
        result = 'Step 3 predictions not found'
    return f"seed {seed} ({'loaded' if loaded else 'trained'}) vs Step 3: {result} | trained with {env_text(env)}"


def mismatch_report(seed, env, model_dir):
    meta_path = run_paths(RESULTS / 'step3', RUN, seed)[1]
    step3_env = json.loads(meta_path.read_text(encoding='utf-8')).get('environment') if meta_path.exists() else None
    return '\n'.join([
        f'STOP: the model of seed {seed} does not reproduce Step 3 (tolerance {STEP3_TOLERANCE:g} %FW), so it is not '
        'analysed.',
        f'  this model trained with: {env_text(env)}',
        f'  Step 3 trained with:     {env_text(step3_env)}',
        f'  The model stays saved in {model_dir}. Send this output to Claude. To continue anyway, run again with '
        '--accept-step3-mismatch.'])


def error_summary(e):
    return {'median': float(np.median(e)), 'p95': float(np.quantile(e, 0.95)), 'max': float(np.max(e))}


def analyse_set(predict, grad, decider, Xr, units, pops, bands, labels, baseline, f_base, ig_steps, n_draws):
    """Attributions, deletion test and stability for the spectra of one test set (TensorFlow-free: predict and grad
    are functions of a numpy array). The IG completeness error of each spectrum, |sum of attributions -
    (F(x) - F(baseline))|, is summarised in %FW (abs_error) and relative to |F(x) - F(baseline)| (rel_error).
    Returns (completeness summary, faithfulness rows, stability rows, records)."""
    base = decider(Xr, units)
    ids, centre, half = base
    attrs = {'IG': integrated_gradients(grad, Xr, baseline, ig_steps), 'GxI': gradient_x_input(grad, Xr, baseline)}
    abs_err, rel_err = completeness_error(attrs['IG'], predict(Xr), f_base)
    completeness = {'abs_error': error_summary(abs_err), 'rel_error': error_summary(rel_err)}
    faith, stab, recs = [], [], []
    for method, A in attrs.items():
        B_spec = band_sums(A, bands)
        _, B = unit_means(B_spec, units)
        order = np.argsort(-np.abs(B), axis=1)
        ranking = {u: list(o) for u, o in zip(ids, order)}
        out = deletion_test(decider, None, Xr, units, None, None, bands, ranking, None, baseline, base, THRESHOLDS,
                            'mean', n_draws, SEED)
        faith += [{'method': method, **r} for r in out]
        stab.append({'method': method, **stability(B_spec, units, pops, seed=SEED)})
        rec = pd.DataFrame({'method': method, 'fruit': ids, 'p': centre, 'half': half,
                            'decision': decide(centre, half, T_MAIN), 'margin': centre - T_MAIN})
        for j in range(len(labels)):
            rec[f'band_{j + 1}'] = B[:, j]
        rec['top_bands'] = [';'.join(labels[k] for k in o[:3]) for o in order]
        recs.append(rec)
    return completeness, faith, stab, pd.concat(recs, ignore_index=True)


def aggregate(df, keys, cols):
    g = df.groupby(keys, sort=False)[cols]
    return pd.concat([g.mean().add_suffix('_mean'), g.std().add_suffix('_sd')], axis=1).reset_index()


def records(t):
    return t.astype(object).where(t.notna(), None).to_dict('records')


def write_summary():
    """Add the 'step4b' entry of results/summary.json from results/step4b.json (after unzipping the Colab results).
    Results of a smoke run are refused; the comparison of the models with Step 3 is printed."""
    detail = json.loads((RESULTS / 'step4b.json').read_text(encoding='utf-8'))
    cfg = detail['config']
    if cfg['smoke'] or cfg['seeds'] != list(SEEDS):
        raise SystemExit(f"results/step4b.json is not a full run (smoke {cfg['smoke']}, seeds {cfg['seeds']}): "
                         'summary.json not changed')
    print('Models vs Step 3 (max |prediction difference| over all spectra; RMSEP of the new / Step 3 predictions):')
    for r in detail['per_seed']:
        print('  ' + check_line(r['seed'], r['loaded'], r['step3_check'], r['train_meta'].get('environment')))
    n_diff = sum(not r['step3_check']['match'] for r in detail['per_seed'])
    if n_diff:
        print(f'WARNING: {n_diff} of {len(SEEDS)} models differ from Step 3 (run with --accept-step3-mismatch)')
    path = RESULTS / 'summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    summary['step4b'] = {'description': 'Step 4b: IG and gradient x input explanations of the CNN (cnn_AT_calonly) '
                                        'decisions; deletion test vs random bands; stability across scans',
                         **{k: detail[k] for k in SUMMARY_KEYS}}
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('step4b entry written to results/summary.json')


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=RESULTS)
    ap.add_argument('--smoke', action='store_true', help='1 seed, 2 epochs, 40 fruit per test set, few draws')
    ap.add_argument('--accept-step3-mismatch', action='store_true',
                    help='analyse models whose predictions differ from Step 3 (otherwise the script stops)')
    ap.add_argument('--time-budget-hours', type=float, default=TIME_BUDGET_HOURS,
                    help='no new seed starts after this many hours; run again to continue with the saved models')
    ap.add_argument('--summary-only', action='store_true', help='local: add results/step4b.json to summary.json')
    args = ap.parse_args()
    if args.summary_only:
        write_summary()
        return
    seeds, ig_steps, n_draws = ((0,), 8, 5) if args.smoke else (SEEDS, IG_STEPS, N_DRAWS)
    model_dir = args.out / 'models'
    for p in remove_partial_files(args.out):
        print(f'Removed {p}, left by an interrupted run', flush=True)

    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    env = session_environment()
    print(json.dumps(env), flush=True)
    to_train = [s for s in seeds if not model_paths(model_dir, s)[0].exists()]
    if to_train and not env['gpu'] and not args.smoke:
        print(f'No GPU in this session: the models of seeds {to_train} are not trained (Step 3 trained them on a GPU). '
              'Switch the runtime to a GPU and run again.', flush=True)
        sys.exit(EXIT_NO_GPU)

    df = load_data()
    masks = set_masks(df)
    raw, wl = df[wavelength_columns(df)].to_numpy(), wavelengths(df)
    cfg = PreprocessConfig()
    wl_trim = wl[(wl >= cfg.wl_min) & (wl <= cfg.wl_max)]
    assert RUNS[RUN]['pretreatment'] == 'AT' and RUNS[RUN]['scaler_fit'] == 'cal' and RUNS[RUN]['train'] == 'cal'
    X_at = features(raw, wl, 'AT')
    scaler = StandardScaler().fit(X_at[masks['Cal']])
    Xs = scaler.transform(X_at).astype('float32')
    y32 = df[TARGET].to_numpy().astype('float32').reshape(-1, 1)
    train_rows = np.flatnonzero(masks['Cal'])
    if args.smoke:
        train_rows = train_rows[:1000]
    base_bands = band_slices(BLOCK_SIZE)
    bands, labels = block_bands(base_bands, N_BLOCKS, BLOCK_SIZE), band_labels(wl_trim, base_bands)
    baseline = np.zeros(Xs.shape[1], dtype='float32')   # training (Cal) mean after standardisation
    fruit_id, pops = df['fruit'].to_numpy(), df['Pop'].to_numpy()
    test_rows = {}
    for s in TEST_SETS:
        rows = np.flatnonzero(masks[s])
        if args.smoke:
            rows = rows[np.isin(fruit_id[rows], np.unique(fruit_id[rows])[:40])]
        test_rows[s] = rows

    per_seed, faith, stab, expl = [], [], [], []
    from src.explain_nn import tf_gradient_fn, tf_predict_fn
    for seed in seeds:
        hours = (time.time() - t0) / 3600
        if hours > args.time_budget_hours:
            print(f'Time budget of {args.time_budget_hours:g} h reached ({hours:.2f} h): stopping before seed {seed}. '
                  'Run again in a new session: the saved models are loaded, not trained again.', flush=True)
            sys.exit(EXIT_TIME_BUDGET)
        t1 = time.time()
        model, meta, loaded = train_or_load(seed, Xs, y32, train_rows, model_dir, env, args.smoke)
        predict, grad = tf_predict_fn(model), tf_gradient_fn(model)
        pred = predict(Xs)
        check = compare_with_step3(df, pred, seed)
        print(check_line(seed, loaded, check, meta.get('environment'))
              + (' [smoke run: a difference is expected]' if args.smoke else ''), flush=True)
        if not check['match'] and not args.smoke and not args.accept_step3_mismatch:
            print(mismatch_report(seed, meta.get('environment'), model_dir), flush=True)
            sys.exit(EXIT_STEP3_MISMATCH)
        fruit = fruit_table(df, pred)
        tuning = fruit[fruit['Set'] == 'Tuning']
        q = conformal_quantile(tuning['p'] - tuning['y'], ALPHA)
        f_base = float(predict(baseline[None])[0])

        def decider(Xr, ur, q=q, predict=predict):
            ids, c = unit_means(predict(Xr), ur)
            return ids, c, np.full(len(ids), q)

        info = {'seed': seed, 'loaded': loaded, 'train_meta': meta, 'half_width': float(q), 'F_baseline': f_base,
                'step3_check': check}
        for s, rows in test_rows.items():
            comp, f_rows, s_rows, rec = analyse_set(predict, grad, decider, Xs[rows], fruit_id[rows], pops[rows], bands,
                                                    labels, baseline, f_base, ig_steps, n_draws)
            info[f'IG_completeness_{s}'] = comp
            faith += [{'seed': seed, 'set': s, **r} for r in f_rows]
            stab += [{'seed': seed, 'set': s, **r} for r in s_rows]
            expl.append(rec.assign(seed=seed, set=s))
        info['seconds'] = time.time() - t1
        per_seed.append(info)
        comp = {s: info[f'IG_completeness_{s}']['abs_error'] for s in test_rows}
        comp_text = ', '.join(f"{s} {c['median']:.1e}/{c['max']:.1e}" for s, c in comp.items())
        print(f"seed {seed}: half-width {q:.3f} %FW, IG completeness |error| median/max (%FW): {comp_text}, "
              f"{info['seconds']:.0f} s", flush=True)

    faith, stab, expl = pd.DataFrame(faith), pd.DataFrame(stab), pd.concat(expl, ignore_index=True)
    fcols = [c for c in faith.columns if c.startswith(('abs_change', 'decision_change', 'p_'))]
    agg_f = aggregate(faith, ['method', 'set', 'strategy', 'm', 'order'], fcols)
    agg_s = aggregate(stab, ['method', 'set'], ['spearman_same_median', 'spearman_other_median', 'jaccard_top_same_mean',
                                                'jaccard_top_other_mean'])
    key = f'decision_change_T{T_MAIN:g}'
    print(f'\nDeletion test, T = {T_MAIN:g}, mean over seeds: % of fruit whose decision changes, top vs random (mean p):')
    for (method, s), g in agg_f.groupby(['method', 'set'], sort=False):
        top, rnd = g[g.order == 'top'].set_index('m'), g[g.order == 'random'].set_index('m')
        print(f'  {method:<4} {s:<9} ' + ' | '.join(
            f"m={m}: {100 * top.loc[m, key + '_mean']:.1f} vs {100 * rnd.loc[m, key + '_mean']:.1f} "
            f"(p={rnd.loc[m, 'p_' + key + '_mean']:.2f})" for m in (1, 2, 3)))
    config = {'run': RUN, 'seeds': list(seeds), 'ig_steps': ig_steps, 'baseline': 'training (Cal) mean = 0 after standardisation',
              'methods': list(METHODS), 'bands': labels, 'n_blocks': N_BLOCKS, 'n_draws': n_draws, 'seed': SEED,
              'thresholds': list(THRESHOLDS), 't_main': T_MAIN, 'alpha': ALPHA, 'smoke': args.smoke,
              'step3_tolerance': STEP3_TOLERANCE, 'accept_step3_mismatch': args.accept_step3_mismatch,
              'time_budget_hours': args.time_budget_hours}
    detail = {'config': config, 'environment': env, 'per_seed': per_seed, 'aggregate_faithfulness': records(agg_f),
              'aggregate_stability': records(agg_s)}
    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / 'step4b.json'
    marker.unlink(missing_ok=True)   # written last: an existing step4b.json marks a complete set of result files
    faith.to_csv(args.out / 'step4b_faithfulness.csv', index=False)
    stab.to_csv(args.out / 'step4b_stability.csv', index=False)
    expl.to_csv(args.out / 'step4b_explanations.csv', index=False)
    write_text_via_partial(marker, json.dumps(detail, indent=2, allow_nan=False))
    print(f'\nResults written to {args.out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()

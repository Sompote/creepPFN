"""Same-protocol machine-learning baselines for the day-10 task.

Every model is fitted per fold on training-source curves only; hyperparameters and conformal
interval widths are chosen on that fold's validation sources; test sources are scored once.
Two input sets:
  tab    : 7 descriptors + time (NU tabular studies; no creep readings of the specimen)
  prefix : 7 descriptors + time + day-10 readings summarised as features, including the
           log-time and Kelvin prefix extrapolations (strongest hand-built features available)
Models: XGBoost (point + native quantile intervals), random forest, MLP.
Intervals for every model: split-conformal on validation curves (scale-normalised residuals)."""
import json, sys, itertools, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'runs/research'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from creep_prior.train import real_records, point_rows, curve_metrics  # noqa: E402
from creep_prior.pipeline import fit_curve, evaluate  # noqa: E402

TRAIN_CUTS = [3, 5, 7, 10, 14, 21, 28]   # multiple cutoffs enlarge the prefix training set (CreepPFN also sees varied context lengths)
SEED = 0


def prefix_feats(r):
    t, y, tq = r['context_times'], r['context_values'], r['query_times']
    s = max(float(np.max(np.abs(y))), 1.)
    xl = np.log1p(t) - np.log1p(t[0]); slope = max(float(np.dot(xl, y) / max(np.dot(xl, xl), 1e-12)), 0.)
    logt = slope * (np.log1p(tq) - np.log1p(t[0]))
    p, _, _ = fit_curve('kelvin_log', t, y); kel = evaluate('kelvin_log', p, tq, t[0])
    n = len(tq)
    F = np.column_stack([np.tile(r['features'], (n, 1)), np.log1p(tq), np.log1p(tq - t[-1]), np.full(n, np.log1p(t[-1])),
                         np.full(n, np.log1p(t[0])), np.full(n, np.log1p(s)), np.full(n, y[-1] / s), np.full(n, len(t)),
                         logt / s, kel / s])
    return F, s


def tab_feats(r, times):
    n = len(times)
    return np.column_stack([np.tile(r['features'], (n, 1)), np.log1p(times), np.full(n, np.log1p(r['context_times'][0]))])


def build(recs, kind, train=False):
    X, Y, S, idx = [], [], [], []
    for k, r in enumerate(recs):
        if kind == 'tab':
            if train:   # tabular studies train on every reading of the training curves
                tt = np.r_[r['context_times'], r['query_times']]; yy = np.r_[r['context_values'], r['targets']]
                X.append(tab_feats(r, tt)); Y.append(yy); S.append(np.ones(len(tt)))
            else:
                X.append(tab_feats(r, r['query_times'])); Y.append(r['targets']); S.append(np.ones(len(r['query_times'])))
        else:
            F, s = prefix_feats(r); X.append(F); Y.append(r['targets'] / s); S.append(np.full(len(F), s))
        idx.append(np.full(len(X[-1]), k))
    return np.vstack(X), np.concatenate(Y), np.concatenate(S), np.concatenate(idx)


def make(name, hp):
    if name == 'xgb':
        return xgb.XGBRegressor(n_estimators=hp['n'], max_depth=hp['d'], learning_rate=hp['lr'], subsample=.8, colsample_bytree=.8,
                                min_child_weight=hp['mcw'], random_state=SEED, n_jobs=4, verbosity=0)
    if name == 'rf':
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=hp['leaf'], max_features=hp['mf'], random_state=SEED, n_jobs=4)
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=hp['h'], alpha=hp['a'], max_iter=600, early_stopping=True,
                                                        random_state=SEED))


GRIDS = dict(xgb=[dict(n=n, d=d, lr=lr, mcw=m) for n, d, lr, m in itertools.product([300, 800], [3, 5, 7], [.03, .1], [1, 10])],
             rf=[dict(leaf=l, mf=f) for l, f in itertools.product([1, 5, 20], [.33, .66, 1.])],
             mlp=[dict(h=h, a=a) for h, a in itertools.product([(64, 64), (128, 128), (256, 128, 64)], [1e-4, 1e-2])])


def nmae_source(recs, pred):
    rows = [dict(g=r['group'], e=np.mean(np.abs(p - r['targets'])) / max(np.mean(np.abs(r['targets'])), 1.)) for r, p in zip(recs, pred)]
    return pd.DataFrame(rows).groupby('g').e.mean().mean()


def split_pred(yhat, S, idx, n, clip0):
    out = []
    for k in range(n):
        v = yhat[idx == k] * S[idx == k]
        out.append(np.maximum(v, 0) if clip0 else v)
    return out


def run_fold(fold):
    tr = {c: real_records(fold, 'train', float(c))[0] for c in TRAIN_CUTS}
    va = real_records(fold, 'validation', 10.)[0]; te = real_records(fold, 'test', 10.)[0]
    rows, info = [], []
    for kind in ('tab', 'prefix'):
        trr = tr[10] if kind == 'tab' else [r for c in TRAIN_CUTS for r in tr[c]]
        Xtr, Ytr, _, _ = build(trr, kind, train=True)
        Xva, Yva, Sva, Iva = build(va, kind); Xte, Yte, Ste, Ite = build(te, kind)
        for name in ('xgb', 'rf', 'mlp'):
            best = None
            for hp in GRIDS[name]:
                m = make(name, hp).fit(Xtr, Ytr)
                sc = nmae_source(va, split_pred(m.predict(Xva), Sva, Iva, len(va), True))
                if best is None or sc < best[0]: best = (sc, hp, m)
            sc, hp, m = best
            pva = split_pred(m.predict(Xva), Sva, Iva, len(va), True); pte = split_pred(m.predict(Xte), Ste, Ite, len(te), True)
            # split-conformal: 90% quantile of |residual| / scale over validation readings, scale = context max (prefix) or 1 (tab)
            sva = [max(np.max(np.abs(r['context_values'])), 1.) for r in va]; ste = [max(np.max(np.abs(r['context_values'])), 1.) for r in te]
            res = np.concatenate([np.abs(p - r['targets']) / s for p, r, s in zip(pva, va, sva)])
            q = np.quantile(res, .9 * (1 + 1 / len(res)))
            preds = [(p, p - q * s, p + q * s) for p, s in zip(pte, ste)]
            rows.append(point_rows(te, preds, f'{name}_{kind}'))
            info.append(dict(fold=fold.name, model=f'{name}_{kind}', val_nmae=sc, hp=str(hp), conformal_q=float(q)))
            if name == 'xgb':   # native quantile intervals with the selected point hyperparameters
                qs = {}
                for a in (.05, .5, .95):
                    mq = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=a, n_estimators=hp['n'], max_depth=hp['d'],
                                          learning_rate=hp['lr'], subsample=.8, colsample_bytree=.8, min_child_weight=hp['mcw'],
                                          random_state=SEED, n_jobs=4, verbosity=0).fit(Xtr, Ytr)
                    qs[a] = split_pred(mq.predict(Xte), Ste, Ite, len(te), False)
                preds = [(np.maximum(m_, 0), np.minimum(lo, m_), np.maximum(hi, m_)) for m_, lo, hi in zip(qs[.5], qs[.05], qs[.95])]
                rows.append(point_rows(te, preds, f'xgbq_{kind}'))
            print(fold.name, name, kind, f'val {100*sc:.2f}', hp, flush=True)
    return pd.concat(rows, ignore_index=True), info


if __name__ == '__main__':
    t0 = time.time(); allp, allinfo = [], []
    for fold in sorted((ROOT / 'data/folds').glob('fold_??')):
        p, i = run_fold(fold); allp.append(p.assign(fold=fold.name)); allinfo += i
    pd.concat(allp).to_csv(OUT / 'ml_predictions.csv', index=False)
    (OUT / 'ml_selection.json').write_text(json.dumps(allinfo, indent=2))
    print(f'done in {time.time()-t0:.0f} s')

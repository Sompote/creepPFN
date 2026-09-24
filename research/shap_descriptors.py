"""Exact five-group interventional SHAP for the final CreepPFN (hierarchical prior, fine-tuned).

Same protocol as the earlier four-group study (SHAP_PROTOCOL.md), extended to the seven-channel model:
groups = density; strength; modulus + missingness flag; stress ratio + missingness flag;
early history + first-reading anchor. All 32 coalitions are enumerated. Background: two panels,
each with one eligible curve drawn uniformly within every training source of the fold.
Explained output: mean of the three member medians (microstrain/MPa) at days 28, 90 and 160.
Each of the 610 test curves is explained only by the fold model that withheld its source."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from creep_prior.model import CreepPFN            # noqa: E402
from creep_prior.train import real_records, pack, tensors, forward  # noqa: E402
from creep_prior.pipeline import sha256           # noqa: E402

GROUPS = ['density', 'strength', 'modulus_and_missingness', 'stress_ratio_and_missingness', 'early_history_and_anchor']
G = len(GROUPS); NC = 2 ** G
HORIZONS = [28., 90., 160.]
# feature order: log_rho, log_fc, log_E28, E28_missing, log1p_anchor_day, stress_ratio, stress_ratio_missing
IDX = {0: [0], 1: [1], 2: [2, 3], 3: [5, 6], 4: [4]}


def exact_values(values):
    out = np.zeros((G,) + values.shape[1:])
    for j in range(G):
        for mask in range(NC):
            if mask & (1 << j): continue
            k = bin(mask).count('1')
            out[j] += math.factorial(k) * math.factorial(G - 1 - k) / math.factorial(G) * (values[mask | (1 << j)] - values[mask])
    return out


def hybrid(target, donor, mask):
    x = np.array(donor['features'], copy=True)
    for j, cols in IDX.items():
        if mask & (1 << j): x[cols] = target['features'][cols]
    h = target if mask & (1 << 4) else donor
    return dict(context_times=h['context_times'], context_values=h['context_values'], features=x,
                query_times=np.asarray(HORIZONS), targets=np.zeros(3))


def predict(models, records, device, bs):
    outs = []
    with torch.inference_mode():
        for s in range(0, len(records), bs):
            b = tensors(pack(records[s:s + bs]), device); vals = []
            for m in models:
                mu, _, scale = forward(m, b); vals.append(torch.sinh(mu) * scale[:, None])
            outs.append(torch.stack(vals).mean(0).cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cuda'); ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    out = ROOT / 'runs/research/shap'; out.mkdir(parents=True, exist_ok=True); dev = torch.device(a.device)
    if dev.type == 'cuda': torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    t0 = time.time(); rows, bg, man = [], [], {}
    for fi in range(1, 6):
        fold = f'fold_{fi:02}'; pdir = ROOT / 'data/folds' / fold
        train, _ = real_records(pdir, 'train'); tgt, _ = real_records(pdir, 'test')
        tgt = sorted(tgt, key=lambda r: r['curve_id'])[:a.limit or None]
        groups = sorted({r['group'] for r in train}); assert not {r['group'] for r in tgt} & set(groups)
        panels = []
        for rep in range(2):
            rng = np.random.default_rng(20260916 + 100 * fi + rep); panel = []
            for g in groups:
                c = sorted([r for r in train if r['group'] == g], key=lambda r: r['curve_id']); d = c[int(rng.integers(len(c)))]
                panel.append(d); bg.append(dict(fold=fold, panel=rep, group=g, curve_id=d['curve_id']))
            panels.append(panel)
        donors = panels[0] + panels[1]; n = len(panels[0])
        models, hashes = [], {}
        for s in (45, 46, 47):
            p = ROOT / 'models' / fold / f'seed{s}.pt'
            w = torch.load(p, map_location='cpu', weights_only=True)
            assert w['training_config']['prior_sha256'] == sha256(pdir / 'prior.json'), 'prior mismatch'
            m = CreepPFN(**w['model_config']).to(dev); m.load_state_dict(w['state_dict']); m.eval(); models.append(m); hashes[s] = sha256(p)
        cv = np.zeros((len(tgt), 2, NC, 3))
        for ti, t in enumerate(tgt):
            recs = [hybrid(t, d, mask) for mask in range(NC) for d in donors]
            cv[ti] = predict(models, recs, dev, a.batch_size).reshape(NC, 2, n, 3).mean(2).transpose(1, 0, 2)
            if (ti + 1) % 25 == 0: print(f'{fold} {ti+1}/{len(tgt)} {time.time()-t0:.0f}s', flush=True)
        direct = predict(models, [hybrid(r, r, NC - 1) for r in tgt], dev, a.batch_size)
        err = float(np.max(np.abs(cv[:, :, NC - 1] - direct[:, None])))
        assert np.allclose(cv[:, :, NC - 1], direct[:, None], rtol=2e-5, atol=2e-4), err
        np.savez_compressed(out / f'{fold}_coalitions.npz', values=cv, curve_ids=np.array([r['curve_id'] for r in tgt]))
        for ti, t in enumerate(tgt):
            for rep in range(2):
                v = cv[ti, rep]; phi = exact_values(v); res = v[NC - 1] - v[0] - phi.sum(0); assert np.max(np.abs(res)) < 1e-8
                for hi, h in enumerate(HORIZONS):
                    r = dict(fold=fold, curve_id=t['curve_id'], source=t['group'], panel=rep, horizon=h, base=v[0, hi], prediction=v[NC - 1, hi],
                             additivity_residual=res[hi]); r.update({g: phi[j, hi] for j, g in enumerate(GROUPS)}); rows.append(r)
        man[fold] = dict(test_curves=len(tgt), training_sources=n, checkpoint_sha256=hashes, direct_max_abs_diff=err)
        pd.DataFrame(rows).to_csv(out / 'attributions.csv', index=False); pd.DataFrame(bg).to_csv(out / 'background.csv', index=False)
        print(f'{fold} complete {time.time()-t0:.0f}s', flush=True)
    (out / 'protocol.json').write_text(json.dumps(dict(groups=GROUPS, horizons=HORIZONS, folds=man, elapsed_seconds=time.time() - t0,
        torch=torch.__version__, gpu=torch.cuda.get_device_name() if dev.type == 'cuda' else None, script_sha256=sha256(__file__)), indent=2))
    (out / 'COMPLETE').write_text('ok\n')


if __name__ == '__main__':
    main()

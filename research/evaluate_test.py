"""Score the 15 packaged networks on the held-out test sources (paper Table 4, CreepPFN row).

Each fold's three members forecast its test curves from readings up to day 10; members are
combined into the fold ensemble (median average, 90% mixture range) and the five folds are pooled.
Writes runs/research/creeppfn_test_predictions.csv, which should reproduce
data/results/creeppfn_test_predictions.csv, and prints the source-macro summary."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'runs/research'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from creep_prior import architecture_study as study  # noqa: E402
from creep_prior.train import summarize  # noqa: E402


def main(device='auto'):
    frames = []
    for fold in sorted((ROOT / 'data/folds').glob('fold_??')):
        members = []
        for seed in (45, 46, 47):
            out = OUT / 'test' / fold.name / f'seed{seed}'
            if not (out / 'description.json').exists():
                study.evaluate_run(SimpleNamespace(fold=fold, checkpoint=ROOT / 'models' / fold.name / f'seed{seed}.pt',
                                                   out=out, label='creeppfn', seed=seed, device=device, threads=8))
            members.append(pd.read_csv(out / 'predictions.csv'))
        frames.append(study.ensemble_points(members, fold, 'creeppfn').assign(fold=fold.name))
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(OUT / 'creeppfn_test_predictions.csv', index=False)
    print(json.dumps(summarize(pooled)['creeppfn'], indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:2])

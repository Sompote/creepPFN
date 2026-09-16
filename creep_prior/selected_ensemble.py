"""Select one configuration per fold on validation and average three seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .pipeline import dump_json
from .train import real_records


ARMS = ("fixed_small", "online_small", "online_medium", "online_large")
TUNABLE = ("online_medium", "online_large")


def select(args):
    root = Path(args.results)
    selection = {}
    for fold in sorted(root.glob("fold_??")):
        candidates = []
        for arm in ARMS:
            base = fold / arm / f"seed{args.seed}"
            paths = [(base, False)]
            if arm in TUNABLE:
                paths.append((base / "real_finetune", True))
            for directory, tuned in paths:
                info = json.loads((directory / "training_summary.json").read_text())
                candidates.append(dict(arm=arm, real_finetune=tuned,
                                       validation_nmae=info["best_validation_nmae"],
                                       checkpoint=str(directory / "best.pt")))
        winner = min(candidates, key=lambda x: (x["validation_nmae"], x["arm"], x["real_finetune"]))
        selection[fold.name] = dict(winner=winner, candidates=candidates,
                                    rule="minimum seed-42 validation source-macro NMAE among six prespecified configurations")
    dump_json(root / "selection_per_fold.json", selection)
    print(json.dumps({k: v["winner"] for k, v in selection.items()}, indent=2), flush=True)


def ensemble(args):
    root = Path(args.results)
    folds = Path(args.folds)
    selection = json.loads((root / "selection_per_fold.json").read_text())
    for fold, info in sorted(selection.items()):
        chosen = info["winner"]
        frames = []
        for seed in args.seeds:
            base = root / fold / chosen["arm"] / f"seed{seed}"
            if chosen["real_finetune"]:
                base /= "real_finetune"
            points = pd.read_csv(base / "test" / "predictions.csv")
            points = points.sort_values(["curve_id", "t_day"]).reset_index(drop=True)
            frames.append(points)
        cols = ("curve_id", "group", "t_day", "observed")
        for frame in frames[1:]:
            if not frames[0][list(cols)].equals(frame[list(cols)]):
                raise ValueError(f"Seed predictions are not aligned in {fold}")
        result = frames[0].copy()
        result["method"] = "selected_ensemble_3seed"
        result["predicted"] = np.mean([f.predicted.to_numpy() for f in frames], axis=0)
        records, _ = real_records(folds / fold, "test", 10.)
        context_scale = {r["curve_id"]: max(float(np.max(np.abs(r["context_values"]))), 1.) for r in records}
        scale = np.array([context_scale[c] for c in result.curve_id], float)
        mu = np.stack([np.arcsinh(f.predicted.to_numpy() / scale) for f in frames])
        zlo = np.stack([np.arcsinh(f.lower90.to_numpy() / scale) for f in frames])
        zhi = np.stack([np.arcsinh(f.upper90.to_numpy() / scale) for f in frames])
        sigma = np.maximum((zhi - zlo) / (2 * 1.6448536269514722), 1e-6)

        def mixture_quantile(prob):
            lower = np.min(mu - 8 * sigma, axis=0)
            upper = np.max(mu + 8 * sigma, axis=0)
            for _ in range(48):
                middle = (lower + upper) / 2
                cdf = ndtr((middle[None, :] - mu) / sigma).mean(axis=0)
                lower = np.where(cdf < prob, middle, lower)
                upper = np.where(cdf >= prob, middle, upper)
            return np.sinh((lower + upper) / 2) * scale

        result["lower90"] = mixture_quantile(.05)
        result["upper90"] = mixture_quantile(.95)
        output = root / fold / "selected_ensemble_3seed" / "test"
        output.mkdir(parents=True, exist_ok=False)
        result.to_csv(output / "predictions.csv", index=False)
        dump_json(output / "description.json", dict(fold=fold, seeds=args.seeds,
                  chosen_configuration=chosen, point_aggregation="arithmetic mean of predictive medians",
                  interval_aggregation="central 90% quantiles of the equally weighted transformed-Gaussian predictive mixture"))
        print(json.dumps({"fold": fold, "curves": result.curve_id.nunique(),
                          "configuration": chosen["arm"], "real_finetune": chosen["real_finetune"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--results", type=Path, required=True)
    s.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("ensemble")
    e.add_argument("--results", type=Path, required=True)
    e.add_argument("--folds", type=Path, required=True)
    e.add_argument("--seeds", type=int, nargs=3, default=(42, 43, 44))
    args = p.parse_args()
    {"select": select, "ensemble": ensemble}[args.command](args)


if __name__ == "__main__":
    main()

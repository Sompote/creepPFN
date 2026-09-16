"""Paired source-level contrasts for the prespecified PFN follow-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import dump_json
from .train import curve_metrics, summarize


CONTRASTS = (
    ("fresh_tasks", "online_small_seed42", "fixed_small_seed42"),
    ("medium_vs_small", "online_medium_seed42", "online_small_seed42"),
    ("large_vs_medium", "online_large_seed42", "online_medium_seed42"),
    ("medium_real_finetune", "online_medium_real_finetune_seed42", "online_medium_seed42"),
    ("large_real_finetune", "online_large_real_finetune_seed42", "online_large_seed42"),
    ("ensemble_vs_fixed_small", "selected_ensemble_3seed", "fixed_small_seed42"),
    ("ensemble_vs_kelvin", "selected_ensemble_3seed", "kelvin_prefix"),
    ("ensemble_vs_log_time", "selected_ensemble_3seed", "log_time"),
    ("ensemble_vs_persistence", "selected_ensemble_3seed", "persistence"),
)


def analyze(root: Path):
    points = pd.read_csv(root / "pooled_predictions.csv")
    curves = curve_metrics(points)
    wide = curves.groupby(["group", "method"]).normalized_mae.mean().unstack("method")
    expected_sources = 65
    contrasts = {}
    rng = np.random.default_rng(20260915)
    for name, intervention, reference in CONTRASTS:
        if intervention not in wide or reference not in wide:
            raise ValueError(f"Missing method for {name}")
        matched = wide[[intervention, reference]].dropna()
        if len(matched) != expected_sources:
            raise ValueError(f"{name} covers {len(matched)} rather than {expected_sources} sources")
        delta = (matched[intervention] - matched[reference]).to_numpy()
        draws = rng.integers(len(delta), size=(2000, len(delta)))
        lo, hi = np.quantile(delta[draws].mean(1), [.025, .975])
        contrasts[name] = dict(intervention=intervention, reference=reference,
                               matched_sources=len(delta), difference_percentage_points=float(delta.mean() * 100),
                               source_bootstrap_95ci_percentage_points=[float(lo * 100), float(hi * 100)])

    selection = json.loads((root / "selection_per_fold.json").read_text())
    seed_metrics = {}
    for seed in (42, 43, 44):
        frames = []
        for fold, info in sorted(selection.items()):
            winner = info["winner"]
            base = root / fold / winner["arm"] / f"seed{seed}"
            if winner["real_finetune"]:
                base /= "real_finetune"
            frame = pd.read_csv(base / "test" / "predictions.csv")
            frame["method"] = f"selected_seed{seed}"
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        seed_metrics[str(seed)] = summarize(combined, bootstrap=0)[f"selected_seed{seed}"]

    pooled = json.loads((root / "pooled_metrics.json").read_text())
    selected = pooled["metrics"]["selected_ensemble_3seed"]
    ci = selected["source_macro_normalized_mae_95ci"]
    report = dict(metric=pooled["metric"], evaluable_curves=pooled["evaluable_outer_test_curves"],
                  sources=pooled["evaluable_outer_test_sources"],
                  prespecified_source_paired_contrasts=contrasts,
                  selected_seed_source_macro_nmae={s: m["source_macro_normalized_mae"] for s, m in seed_metrics.items()},
                  selected_seed_nmae_sd=float(np.std([m["source_macro_normalized_mae"] for m in seed_metrics.values()], ddof=1)),
                  selected_ensemble_nmae=selected["source_macro_normalized_mae"],
                  selected_ensemble_source_bootstrap_95ci=ci,
                  target_10_percent=dict(point_estimate_below=selected["source_macro_normalized_mae"] < .1,
                                         source_bootstrap_interval_entirely_below=ci[1] < .1),
                  uncertainty="Source bootstrap conditions on the fitted models; three seed values describe finite-seed variation but are not a confidence interval",
                  scope="Held-out literature sources within the curated NU cohort; no external-cohort estimate")
    dump_json(root / "followup_analysis.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    analyze(args.results)


if __name__ == "__main__":
    main()

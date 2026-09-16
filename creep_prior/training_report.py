"""Summarize a completed, frozen-checkpoint experiment without selecting models."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "creep-prior-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.evaluation / "metrics.json").read_text())
    config = json.loads((args.run / "config.json").read_text())
    summary = json.loads((args.run / "training_summary.json").read_text())
    history = pd.read_csv(args.run / "history.csv")
    points = pd.read_csv(args.evaluation / "predictions.csv")
    raw = pd.read_csv(args.prior_dir / "real_observations.csv")
    labels = {"creep_pfn": "Creep PFN", "persistence": "Persistence", "log_time": "Log-time fit", "kelvin_prefix": "Kelvin prefix fit"}
    methods = ["creep_pfn", "persistence", "log_time", "kelvin_prefix"]
    rows = []
    for method in methods:
        m = report["metrics"][method]
        lo, hi = m["source_macro_normalized_mae_95ci"]
        rows.append(f"| {labels[method]} | {m['source_macro_mae']:.3f} | {100*m['source_macro_normalized_mae']:.2f}% | {100*lo:.2f}–{100*hi:.2f}% |")
    pfn = report["metrics"]["creep_pfn"]
    lines = ["# Initial synthetic-pretrained creep predictor", "",
        "## Experimental protocol", "",
        f"A {config['parameters']:,}-parameter conditional transformer was trained on {config['synthetic_tasks']:,} synthetic tasks. "
        "Its weights received gradients only from synthetic noisy targets. The synthetic prior was calibrated on real training sources; "
        "this is not a model developed independently of experimental data.", "",
        f"Training used seed {config['seed']} and completed {summary['completed_epochs']} epochs in {summary['elapsed_seconds']/60:.2f} minutes. "
        f"Checkpoint epoch {report['checkpoint_epoch']} was selected by source-macro normalized MAE on "
        f"{config['validation_curves']} real validation curves from {config['validation_sources']} sources. "
        f"The frozen checkpoint was evaluated on {pfn['curves']} test curves from {pfn['sources']} sources "
        f"({pfn['observations']} future observations). No test-based tuning was performed.", "",
        f"Excluded at the fixed cutoff because of insufficient context or future readings: "
        f"{', '.join(report['excluded_curves']) if report['excluded_curves'] else 'none'}. "
        "The protocol requires at least two context readings and at least one later measurement.", "",
        f"All context readings occur on or before day {report['cutoff_day']:g}; targets are later original observations through day 160. "
        "Compliance is expressed as change after the first observation, in µε/MPa. "
        "Properties are density, strength and independently recorded E28, with a missingness indicator. "
        "No full-curve modulus estimates, daily fitted targets or future target values enter the predictor.", "",
        "## Test results", "",
        "| Method | Source-macro MAE (µε/MPa) | Source-macro normalized MAE | 95% source-bootstrap interval |",
        "|---|---:|---:|---:|", *rows, "",
        "For each curve, normalized MAE is its mean absolute future error divided by "
        "max(mean absolute measured future increment, 1 µε/MPa). Curve scores are averaged within source, "
        "then sources receive equal weight. This is not pointwise MAPE and is not directly comparable "
        "to the absolute-compliance MAPE in the original manuscript. Confidence intervals use 2,000 "
        "percentile bootstrap replicates over source means; they do not measure training-seed variation.", "",
        "## Predictive intervals", "",
        f"The model's nominal 90% marginal intervals attained {100*pfn['source_macro_coverage90']:.2f}% "
        f"source-macro coverage, with mean width {pfn['source_macro_interval_width90']:.3f} µε/MPa. "
        "Intervals are Gaussian in asinh(increment/context-scale) space and transformed back using sinh. "
        "They have not been recalibrated on real data and do not define a joint distribution over entire curves.", "",
        "## Limitations", "",
        "This is an initial run with one model seed, one source partition, two synthetic curve families, "
        "and 10,000 fixed tasks. It does not establish a foundation-model capability or superiority to "
        "well-tuned existing models. The inherited cohort selection and synthetic-prior limitations remain. "
        "Output curves are not constrained to be nonnegative or monotone. Baselines receive the same "
        "observed prefixes; the log-time fit uses one nonnegative coefficient and the Kelvin fit uses "
        "the generator's fixed basis and penalty, fitted only to context values. Neither is tuned on test data.", "",
        "## Artifacts", "", "- Checkpoint: `best.pt` in the training directory.",
        "- Training history: `history.csv`.", "- Test predictions and per-curve errors: evaluation directory.",
        "- Paired source-bootstrap differences: `metrics.json`.", "- Figures: `training_and_test.pdf` and `.png`.", ""]
    interpretation = ["## Paired baseline comparisons", "",
        "Differences below are PFN minus baseline normalized MAE, in percentage points. "
        "Negative differences favor the PFN. Each bootstrap draw resamples the same sources for both methods.", "",
        "| Baseline | Difference | 95% paired source-bootstrap interval |",
        "|---|---:|---:|"]
    unresolved = []
    for method, result in report['paired_nmae_difference_pfn_minus_baseline'].items():
        low, high = result['ci95']
        interpretation.append(f"| {labels[method]} | {100*result['difference']:.2f} | {100*low:.2f} to {100*high:.2f} |")
        if low <= 0 <= high:
            unresolved.append(labels[method])
    if unresolved:
        interpretation.extend(["", "The comparison is statistically unresolved at this bootstrap interval level for: "
                               + ", ".join(unresolved) + ". A lower point estimate alone does not establish superiority."])
    lines.extend(interpretation + [""])
    (args.evaluation / "RESULTS.md").write_text("\n".join(lines))
    # Horizon-specific metrics: recompute curve and source means within each bin.
    horizon_rows = []
    for left, right in [(10, 28), (28, 60), (60, 90), (90, 160)]:
        subset = points[(points.t_day > left) & (points.t_day <= right)].copy()
        subset["abs_error"] = abs(subset.predicted - subset.observed)
        for method, p in subset.groupby("method"):
            curve = p.groupby(["group", "curve_id"]).abs_error.mean()
            group = curve.groupby("group").mean()
            horizon_rows.append(dict(method=method, after_day=left, through_day=right,
                                     observations=len(p), curves=len(curve), sources=len(group), source_macro_mae=group.mean()))
    pd.DataFrame(horizon_rows).to_csv(args.evaluation / "errors_by_horizon.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    axes[0, 0].plot(history.epoch, history.validation_nmae * 100, color="#286d9b")
    axes[0, 0].axvline(summary["best_epoch"], color="gray", linestyle="--")
    axes[0, 0].set(title="Checkpoint selection on validation sources", xlabel="Epoch", ylabel="Source-macro normalized MAE (%)")
    values = [100 * report['metrics'][m]['source_macro_normalized_mae'] for m in methods]
    cis = np.array([report['metrics'][m]['source_macro_normalized_mae_95ci'] for m in methods]) * 100
    axes[0, 1].bar(range(4), values, color=["#286d9b", "#aaa", "#d6893b", "#779779"])
    axes[0, 1].errorbar(range(4), values, yerr=np.maximum(0, np.vstack([np.array(values)-cis[:, 0], cis[:, 1]-values])), fmt="none", color="black", capsize=3)
    axes[0, 1].set_xticks(range(4), [labels[m] for m in methods], rotation=20, ha="right")
    axes[0, 1].set(title="Held-out sources: 95% bootstrap intervals", ylabel="Source-macro normalized MAE (%)")
    ids = np.array(sorted(points.curve_id.unique()))
    selected = ids[np.linspace(0, len(ids) - 1, 4, dtype=int)]
    for ax, cid in zip([axes[0, 2], *axes[1]], selected):
        pred = points[(points.curve_id == cid) & (points.method == "creep_pfn")].sort_values("t_day")
        context = raw[(raw.curve_id == cid) & (raw.t_day <= config["cutoff_day"])]
        ax.scatter(context.t_day, context.increment_ue_per_MPa, color="black", s=12, label="Context")
        ax.scatter(pred.t_day, pred.observed, color="#d6893b", marker="x", s=20, label="Measured future")
        ax.plot(pred.t_day, pred.predicted, color="#286d9b", label="PFN median")
        ax.fill_between(pred.t_day, pred.lower90, pred.upper90, color="#286d9b", alpha=.15, label="90% interval")
        ax.axvline(config["cutoff_day"], color="gray", linestyle="--", lw=.8)
        ax.set(title=f"{cid} ({pred.group.iloc[0]})", xlabel="Day after loading", ylabel="Compliance increment (µε/MPa)")
    axes[0, 2].legend(fontsize=7)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Initial creep PFN: one training seed; test examples selected by evenly spaced curve IDs", fontsize=13)
    fig.savefig(args.evaluation / "training_and_test.png", dpi=160)
    fig.savefig(args.evaluation / "training_and_test.pdf")
    plt.close(fig)
    print(args.evaluation / "RESULTS.md")


if __name__ == "__main__":
    main()

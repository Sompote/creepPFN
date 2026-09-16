"""Create descriptive plots using synthetic tasks and training fits only."""
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

from .pipeline import FAMILIES, evaluate, dump_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.tasks, allow_pickle=False)
    fits = pd.read_csv(args.prior_dir / "training_curve_fits.csv")
    manifest = pd.read_csv(args.prior_dir / "split_manifest.csv")
    train_ids = set(manifest.loc[manifest.split == "train", "curve_id"])
    assert set(data["donor_curve_id"]).issubset(train_ids)
    assert set(fits.curve_id).issubset(train_ids)
    colors = {"weibull": "#2268a2", "kelvin_log": "#d07828"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    magnitude = np.array([p[0] if f == "weibull" else p.sum()
                          for f, p in zip(data["family"], data["parameters"])])
    rng = np.random.default_rng(7)
    selected = rng.choice(len(magnitude), min(20, len(magnitude)), replace=False)
    for i in selected:
        f = data["family"][i]
        t = np.linspace(data["anchor_day"][i], 160, 200)
        p = data["parameters"][i, :3 if f == "weibull" else 5]
        axes[0, 0].plot(t, evaluate(f, p, t, t[0]), color=colors[f], alpha=.45, lw=1.2)
    axes[0, 0].set(title="A. Sampled latent curves (20 tasks)", xlabel="Elapsed time after loading (days)",
                   ylabel="Compliance increment (µε/MPa)")
    fitted = {f: np.array([json.loads(p)[0] if f == "weibull" else sum(json.loads(p))
                           for p in fits.loc[fits.family == f, "params"]]) for f in FAMILIES}
    bins = np.geomspace(min(magnitude.min(), *(v.min() for v in fitted.values())) * .9,
                       max(magnitude.max(), *(v.max() for v in fitted.values())) * 1.1, 35)
    for f in FAMILIES:
        axes[0, 1].hist(magnitude[data["family"] == f], bins=bins, density=True,
                        histtype="step", color=colors[f], label=f"Synthetic {f}")
    axes[0, 1].hist(fitted["weibull"], bins=bins, density=True, histtype="step", color="black",
                    linestyle="--", label="Training Weibull fits (per curve)")
    axes[0, 1].set(xscale="log", title="B. Day-160 magnitude: descriptive comparison",
                   xlabel="Compliance increment at day 160 (µε/MPa)", ylabel="Density")
    axes[0, 1].legend(fontsize=8)
    for f in FAMILIES:
        indices = np.flatnonzero(data["family"] == f)[::10]
        axes[1, 0].scatter(data["properties"][indices, 1], magnitude[indices], s=8,
                           alpha=.2, color=colors[f], label=f)
    axes[1, 0].set(title="C. Generated property–response pairs", xlabel="Compressive strength (MPa)",
                   ylabel="Day-160 compliance increment (µε/MPa)", yscale="log")
    axes[1, 0].legend(fontsize=8)
    i = int(selected[0])
    ctx, target = data["context_mask"][i], data["target_mask"][i]
    for mask, label, marker, color in [(ctx, "Context supplied to model", "o", "#2268a2"),
                                      (target, "Withheld training targets", "x", "#d07828")]:
        axes[1, 1].scatter(data["times_day"][i, mask], data["observed_increment"][i, mask],
                           marker=marker, label=label, color=color)
    axes[1, 1].axvline(data["cutoff_day"][i], color="gray", linestyle="--", lw=1)
    axes[1, 1].set(title="D. One synthetic prediction task", xlabel="Elapsed time after loading (days)",
                   ylabel="Observed compliance increment (µε/MPa)")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Synthetic creep prior: development diagnostics, not forecast validation", fontsize=13)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.out / "synthetic_diagnostics.png", dpi=160)
    fig.savefig(args.out / "synthetic_diagnostics.pdf")
    plt.close(fig)
    report = {"donors_from_training_only": True, "fits_from_training_only": True,
              "comparison_note": "Synthetic sources sampled uniformly; training histogram weights curves equally.",
              "families": {}}
    for f in FAMILIES:
        report["families"][f] = dict(
            quantile_levels=[.05, .5, .95],
            training_fit_M160=np.quantile(fitted[f], [.05, .5, .95]).tolist(),
            synthetic_M160=np.quantile(magnitude[data["family"] == f], [.05, .5, .95]).tolist())
    dump_json(args.out / "diagnostics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

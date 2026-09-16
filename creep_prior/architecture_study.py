"""Nested five-fold architecture ablations and validation-only hyperparameter search.

Tune one factor at a time around the published model using seed 42 and the
existing disjoint validation sources in each outer fold. Retrain the selected
configuration at seeds 43 and 44. Retrain each architecture removal at the
fold-selected hyperparameters for seeds 42--44. Outer-test observations are
loaded only by the separate evaluate phase.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr

from .cv_experiment import checkpoint, finetune, score, task_records, train_epoch
from .model import CreepPFN
from .pipeline import dump_json, sample_tasks, sha256
from .train import choose_device, curve_metrics, point_rows, predict, real_records, summarize

BASE = dict(width=256, layers=4, heads=4, dropout=.05)
TUNING = {
    "reference": (dict(BASE), 3e-4, 1e-5),
    "width128": (dict(BASE, width=128), 3e-4, 1e-5),
    "depth2": (dict(BASE, layers=2), 3e-4, 1e-5),
    "depth6": (dict(BASE, layers=6), 3e-4, 1e-5),
    "heads8": (dict(BASE, heads=8), 3e-4, 1e-5),
    "dropout0": (dict(BASE, dropout=0.), 3e-4, 1e-5),
    "dropout15": (dict(BASE, dropout=.15), 3e-4, 1e-5),
    "pre_lr_low": (dict(BASE), 1e-4, 1e-5),
    "pre_lr_high": (dict(BASE), 6e-4, 1e-5),
    "fine_lr_low": (dict(BASE), 3e-4, 3e-6),
    "fine_lr_high": (dict(BASE), 3e-4, 3e-5),
}
ABLATIONS = {
    "no_context_attention": dict(self_attention=False),
    "no_query_attention": dict(cross_attention=False),
    "no_attention": dict(self_attention=False, cross_attention=False),
    "no_properties": dict(use_properties=False),
    "no_query_gap": dict(use_query_gap=False),
    "no_query_skip": dict(query_skip=False),
    "linear_query": dict(query_mlp=False),
    "fixed_scale": dict(learn_scale=False),
}
SEEDS = (42, 43, 44)


def fit_run(args):
    fold, root = Path(args.fold), Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model_config = json.loads(args.model_config)
    model = CreepPFN(**model_config).to(choose_device(args.device))
    device = next(model.parameters()).device
    prior = json.loads((fold / "prior.json").read_text())
    validation, excluded = real_records(fold, "validation", 10.)
    if not validation:
        raise ValueError("No eligible validation curves")
    pre = root / "pretrain"
    pre.mkdir()
    config = dict(arm=args.label, fold=fold.name, seed=args.seed,
                  model=model.config, parameters=sum(p.numel() for p in model.parameters()),
                  tasks_per_epoch=args.tasks, epochs=args.epochs, patience=args.patience,
                  batch_size=128, lr=args.pre_lr, fine_lr=args.fine_lr,
                  weight_decay=.001, cutoff_day=10.,
                  prior_sha256=sha256(fold / "prior.json"),
                  split_sha256=sha256(fold / "split_manifest.csv"),
                  generator_sha256=sha256(Path(__file__).parent / "pipeline.py"),
                  model_sha256=sha256(Path(__file__).parent / "model.py"),
                  study_sha256=sha256(__file__),
                  validation_curves=len(validation),
                  validation_sources=len(set(r["group"] for r in validation)),
                  excluded_validation_curves=excluded,
                  device=str(device), torch_version=str(torch.__version__),
                  selection="source-macro NMAE on disjoint validation sources only",
                  gradients="synthetic targets during pretraining")
    dump_json(pre / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pre_lr, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.pre_lr / 10)
    rng = np.random.default_rng(args.seed)
    best, best_epoch, history = float("inf"), 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        tasks = task_records(sample_tasks(prior, args.tasks, args.seed + 10_000 + epoch)[0])
        nll = train_epoch(model, optimizer, tasks, device, 128, rng)
        scheduler.step()
        val = score(model, validation, device, 128)
        if val < best:
            best, best_epoch = val, epoch
            checkpoint(model, pre, epoch, val, config)
        history.append(dict(epoch=epoch, synthetic_nll=nll,
                            validation_source_macro_nmae=val,
                            best_epoch=best_epoch,
                            elapsed_seconds=time.monotonic() - started))
        pd.DataFrame(history).to_csv(pre / "history.csv", index=False)
        print(json.dumps(dict(epoch=epoch, val_nmae=val, best=best)), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    dump_json(pre / "training_summary.json", dict(
        best_epoch=best_epoch, best_validation_nmae=best,
        completed_epochs=len(history), elapsed_seconds=time.monotonic() - started,
        checkpoint_sha256=sha256(pre / "best.pt"), test_evaluated=False))
    fine = root / "finetune"
    finetune(SimpleNamespace(fold=fold, out=fine, pretrained=pre / "best.pt",
                             seed=args.seed, epochs=20, patience=5,
                             lr=args.fine_lr, batch_size=64, cutoff=10.,
                             device=args.device, threads=args.threads))
    fine_info = json.loads((fine / "training_summary.json").read_text())
    dump_json(root / "summary.json", dict(
        fold=fold.name, label=args.label, seed=args.seed,
        model_config=model.config, pre_lr=args.pre_lr, fine_lr=args.fine_lr,
        pre_validation_nmae=best,
        best_validation_nmae=fine_info["best_validation_nmae"],
        selected_checkpoint_sha256=sha256(fine / "best.pt"),
        outer_test_evaluated=False))


def evaluate_run(args):
    fold, checkpoint_path, out = Path(args.fold), Path(args.checkpoint), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ckpt["training_config"]
    if sha256(fold / "prior.json") != config["prior_sha256"]:
        raise ValueError("Checkpoint prior does not match outer fold")
    model = CreepPFN(**ckpt["model_config"]).to(choose_device(args.device))
    model.load_state_dict(ckpt["state_dict"])
    records, excluded = real_records(fold, "test", 10.)
    rows = point_rows(records, predict(model, records, next(model.parameters()).device),
                      args.label)
    rows.to_csv(out / "predictions.csv", index=False)
    dump_json(out / "description.json", dict(
        fold=fold.name, label=args.label, seed=args.seed,
        checkpoint_sha256=sha256(checkpoint_path), eligible_curves=len(records),
        excluded_curves=excluded, test_source_groups=len(set(r["group"] for r in records)),
        source_macro_nmae=summarize(rows, bootstrap=0)[args.label]["source_macro_normalized_mae"]))


def worker(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        print(log.read_text()[-4000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"Job failed ({process.returncode}); see {log}")


def run_fit(root, fold, label, model, pre_lr, fine_lr, seed, args):
    out = root / fold.name / label / f"seed{seed}"
    if (out / "summary.json").exists():
        return out
    if out.exists():
        raise FileExistsError(f"Incomplete run: {out}")
    cmd = [sys.executable, "-m", "creep_prior.architecture_study", "fit-run",
           "--fold", str(fold), "--out", str(out), "--label", label,
           "--model-config", json.dumps(model), "--pre-lr", str(pre_lr),
           "--fine-lr", str(fine_lr), "--seed", str(seed),
           "--tasks", str(args.tasks), "--epochs", str(args.epochs),
           "--patience", str(args.patience), "--device", args.device,
           "--threads", str(args.threads)]
    print(json.dumps(dict(phase="fit", fold=fold.name, label=label, seed=seed)), flush=True)
    worker(cmd, root / "logs" / f"fit_{fold.name}_{label}_seed{seed}.log")
    return out


def tune(args):
    folds, root = sorted(Path(args.folds).glob("fold_??")), Path(args.results)
    if len(folds) != 5:
        raise ValueError("Expected five outer folds")
    root.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        candidates = []
        for name, (model, pre_lr, fine_lr) in TUNING.items():
            out = run_fit(root, fold, f"tune_{name}", model, pre_lr, fine_lr, 42, args)
            summary = json.loads((out / "summary.json").read_text())
            candidates.append(dict(name=name, model_config=model,
                                   pre_lr=pre_lr, fine_lr=fine_lr,
                                   validation_nmae=summary["best_validation_nmae"],
                                   checkpoint=str(out / "finetune" / "best.pt")))
        winner = min(candidates, key=lambda x: (x["validation_nmae"], x["name"]))
        dump_json(root / fold.name / "tuning_selection.json", dict(
            fold=fold.name, winner=winner, candidates=candidates,
            rule="minimum seed-42 source-macro NMAE on this fold's disjoint validation sources; no outer-test access"))
        print(json.dumps(dict(fold=fold.name, tuning_winner=winner)), flush=True)
    dump_json(root / "tuning_summary.json", dict(
        search_space=TUNING, outer_folds=5, candidate_count=len(TUNING),
        seed=42, selection="independently within each outer fold on validation sources only",
        phase="complete; outer-test outcomes not evaluated"))


def ablate(args):
    folds, root = sorted(Path(args.folds).glob("fold_??")), Path(args.results)
    for fold in folds:
        selected = json.loads((root / fold.name / "tuning_selection.json").read_text())["winner"]
        model, pre_lr, fine_lr = selected["model_config"], selected["pre_lr"], selected["fine_lr"]
        for seed in (43, 44):
            run_fit(root, fold, f"tune_{selected['name']}", model, pre_lr, fine_lr, seed, args)
        for label, flags in ABLATIONS.items():
            variant = dict(model, **flags)
            for seed in SEEDS:
                run_fit(root, fold, f"ablate_{label}", variant, pre_lr, fine_lr, seed, args)
        print(json.dumps(dict(fold=fold.name, architecture_ablations_complete=True)), flush=True)
    dump_json(root / "fit_complete.json", dict(
        folds=5, tuned_full_seeds=SEEDS, architecture_removals=list(ABLATIONS),
        source="fold-specific synthetic prior and real training sources",
        selection="fold-specific disjoint validation sources; outer-test data not loaded"))


def evaluate(args):
    folds, root = sorted(Path(args.folds).glob("fold_??")), Path(args.results)
    if not (root / "fit_complete.json").exists():
        raise FileNotFoundError("Fit and validation selection must complete before outer-test evaluation")
    for fold in folds:
        selected = json.loads((root / fold.name / "tuning_selection.json").read_text())["winner"]
        labels = ["full"] + list(ABLATIONS)
        for label in labels:
            fit_label = f"tune_{selected['name']}" if label == "full" else f"ablate_{label}"
            for seed in SEEDS:
                base = root / fold.name / fit_label / f"seed{seed}"
                checkpoint_path = base / "finetune" / "best.pt"
                out = root / fold.name / "test" / label / f"seed{seed}"
                if (out / "description.json").exists():
                    continue
                if out.exists():
                    raise FileExistsError(f"Incomplete evaluation: {out}")
                cmd = [sys.executable, "-m", "creep_prior.architecture_study", "evaluate-run",
                       "--fold", str(fold), "--checkpoint", str(checkpoint_path),
                       "--out", str(out), "--label", label, "--seed", str(seed),
                       "--device", args.device, "--threads", str(args.threads)]
                worker(cmd, root / "logs" / f"test_{fold.name}_{label}_seed{seed}.log")
                print(json.dumps(dict(phase="outer_test", fold=fold.name,
                                      label=label, seed=seed)), flush=True)
    dump_json(root / "evaluation_complete.json", dict(
        folds=5, labels=["full"] + list(ABLATIONS), seeds=SEEDS,
        protocol="each model selected on disjoint validation sources before outer-test evaluation"))


def ensemble_points(frames, fold_dir, label):
    frames = [f.sort_values(["curve_id", "t_day"]).reset_index(drop=True) for f in frames]
    cols = ["curve_id", "group", "t_day", "observed"]
    if any(not frames[0][cols].equals(f[cols]) for f in frames[1:]):
        raise ValueError(f"Misaligned seed predictions: {fold_dir.name}/{label}")
    result = frames[0].copy()
    result["method"] = label
    result["predicted"] = np.mean([f.predicted.to_numpy() for f in frames], axis=0)
    records, _ = real_records(fold_dir, "test", 10.)
    scale_map = {r["curve_id"]: max(float(np.max(np.abs(r["context_values"]))), 1.)
                 for r in records}
    scale = np.array([scale_map[c] for c in result.curve_id], float)
    mu = np.stack([np.arcsinh(f.predicted.to_numpy() / scale) for f in frames])
    lo = np.stack([np.arcsinh(f.lower90.to_numpy() / scale) for f in frames])
    hi = np.stack([np.arcsinh(f.upper90.to_numpy() / scale) for f in frames])
    sigma = np.maximum((hi - lo) / (2 * 1.6448536269514722), 1e-6)
    for prob, column in ((.05, "lower90"), (.95, "upper90")):
        lower = np.min(mu - 8 * sigma, axis=0)
        upper = np.max(mu + 8 * sigma, axis=0)
        for _ in range(48):
            middle = (lower + upper) / 2
            cdf = ndtr((middle[None, :] - mu) / sigma).mean(axis=0)
            lower = np.where(cdf < prob, middle, lower)
            upper = np.where(cdf >= prob, middle, upper)
        result[column] = np.sinh((lower + upper) / 2) * scale
    result["fold"] = fold_dir.name
    return result


def analyze(args):
    folds, root = sorted(Path(args.folds).glob("fold_??")), Path(args.results)
    if not (root / "evaluation_complete.json").exists():
        raise FileNotFoundError("Outer-test evaluations are incomplete")
    frames = []
    labels = ["full"] + list(ABLATIONS)
    for fold in folds:
        for label in labels:
            points = [pd.read_csv(root / fold.name / "test" / label / f"seed{seed}" / "predictions.csv")
                      for seed in SEEDS]
            frames.append(ensemble_points(points, fold, label))
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(root / "pooled_predictions.csv", index=False)
    curves = curve_metrics(pooled)
    curves.to_csv(root / "pooled_curve_metrics.csv", index=False)
    metrics = summarize(pooled)
    wide = curves.groupby(["group", "method"]).normalized_mae.mean().unstack("method")
    if wide.shape[0] != 65 or any(wide[label].isna().any() for label in labels):
        raise ValueError("Missing source or method in paired analysis")
    rng = np.random.default_rng(20260915)
    ids = rng.integers(len(wide), size=(2000, len(wide)))
    paired = {}
    for label in ABLATIONS:
        delta = (wide[label] - wide["full"]).to_numpy()
        paired[label] = dict(difference_pctpt=float(delta.mean() * 100),
                             ci95_pctpt=(np.quantile(delta[ids].mean(axis=1), [.025, .975]) * 100).tolist(),
                             matched_sources=len(delta))
    selections = {fold.name: json.loads((root / fold.name / "tuning_selection.json").read_text())
                  for fold in folds}
    dump_json(root / "results.json", dict(
        metric="source-macro mean of curve MAE / max(mean(abs(future increment)), 1 ue/MPa)",
        folds=5, sources=65, eligible_curves=610, future_readings=5646,
        selection="fold-specific seed-42 validation source-macro NMAE; outer test accessed after fit_complete",
        hyperparameter_candidates=TUNING, tuning_selections=selections,
        ablations=ABLATIONS, model_results=metrics,
        paired_ablation_minus_full=paired,
        uncertainty="2000 paired source bootstrap draws; fixed three seeds; no hyperparameter-search uncertainty included"))
    table = pd.DataFrame([
        dict(arm=label, source_macro_nmae_pct=metrics[label]["source_macro_normalized_mae"] * 100,
             source_macro_mae=metrics[label]["source_macro_mae"],
             source_macro_coverage90=metrics[label]["source_macro_coverage90"],
             difference_vs_full_pctpt=paired.get(label, {}).get("difference_pctpt", 0.))
        for label in labels])
    table.to_csv(root / "comparison.csv", index=False)
    print(table.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("fit-run")
    p.add_argument("--fold", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--pre-lr", type=float, required=True)
    p.add_argument("--fine-lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tasks", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    q = sub.add_parser("evaluate-run")
    q.add_argument("--fold", type=Path, required=True)
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--label", required=True)
    q.add_argument("--seed", type=int, required=True)
    for name in ("tune", "ablate", "evaluate", "analyze"):
        s = sub.add_parser(name)
        s.add_argument("--folds", type=Path, required=True)
        s.add_argument("--results", type=Path, required=True)
        s.add_argument("--tasks", type=int, default=5000)
        s.add_argument("--epochs", type=int, default=25)
        s.add_argument("--patience", type=int, default=6)
    for s in (p, q, *[sub.choices[name] for name in ("tune", "ablate", "evaluate", "analyze")]):
        s.add_argument("--device", default="auto")
        s.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    {"fit-run": fit_run, "evaluate-run": evaluate_run,
     "tune": tune, "ablate": ablate, "evaluate": evaluate, "analyze": analyze}[args.phase](args)


if __name__ == "__main__":
    main()

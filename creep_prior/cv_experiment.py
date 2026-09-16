"""Controlled source-disjoint cross-validation of synthetic PFN pretraining.

Fit commands never load outer-test observations. Evaluation is a separate step.
The fixed arm repeats one synthetic shard; online arms draw new tasks per epoch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from .model import CreepPFN, task_nll
from .pipeline import dump_json, sample_tasks, sha256
from .train import (baselines, choose_device, curve_metrics, forward, pack,
                    point_rows, predict, real_records, summarize, tensors)


ARMS = {"fixed_small": (64, 2, False), "online_small": (64, 2, True),
        "online_medium": (128, 4, True), "online_large": (256, 4, True)}


def task_records(data):
    return [dict(context_times=data["times_day"][i, data["context_mask"][i]],
                 context_values=data["observed_increment"][i, data["context_mask"][i]],
                 query_times=data["times_day"][i, data["target_mask"][i]],
                 targets=data["observed_increment"][i, data["target_mask"][i]],
                 features=data["features"][i]) for i in range(len(data["features"]))]


def score(model, records, device, batch_size):
    points = point_rows(records, predict(model, records, device, batch_size), "creep_pfn")
    return summarize(points, bootstrap=0)["creep_pfn"]["source_macro_normalized_mae"]


def train_epoch(model, optimizer, records, device, batch_size, rng):
    model.train()
    losses = []
    ids = rng.permutation(len(records))
    for start in range(0, len(records), batch_size):
        batch = tensors(pack([records[int(i)] for i in ids[start:start + batch_size]]), device)
        optimizer.zero_grad(set_to_none=True)
        mu, sigma, scale = forward(model, batch)
        loss = task_nll(mu, sigma, scale, batch["targets"], batch["target_mask"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append((float(loss.detach().cpu()), len(batch["features"])))
    return float(np.average([v for v, _ in losses], weights=[n for _, n in losses]))


def checkpoint(model, out, epoch, val_score, config):
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    model_config=model.config, epoch=epoch, validation_score=val_score,
                    training_config=config), out / "best.pt")


def fit(args):
    fold = Path(args.fold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    width, layers, online = ARMS[args.arm]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    prior = json.loads((fold / "prior.json").read_text())
    validation, excluded = real_records(fold, "validation", args.cutoff)
    if not validation:
        raise ValueError("No eligible validation curves")
    model = CreepPFN(width, layers, 4, .05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    config = dict(arm=args.arm, fold=fold.name, seed=args.seed, model=model.config,
                  parameters=sum(p.numel() for p in model.parameters()),
                  task_regime="fresh per epoch" if online else "fixed shard repeated",
                  tasks_per_epoch=args.tasks, epochs=args.epochs, patience=args.patience,
                  batch_size=args.batch_size, lr=args.lr, cutoff_day=args.cutoff,
                  prior_sha256=sha256(fold / "prior.json"),
                  split_sha256=sha256(fold / "split_manifest.csv"),
                  generator_sha256=sha256(Path(__file__).parent / "pipeline.py"),
                  model_sha256=sha256(Path(__file__).parent / "model.py"),
                  experiment_sha256=sha256(__file__),
                  training_utilities_sha256=sha256(Path(__file__).parent / "train.py"),
                  validation_curves=len(validation),
                  validation_sources=len(set(r["group"] for r in validation)),
                  excluded_validation_curves=excluded, torch_version=str(torch.__version__),
                  numpy_version=np.__version__, pandas_version=pd.__version__,
                  device=str(device), selection="source-macro NMAE on validation sources only",
                  gradients="synthetic noisy targets only")
    dump_json(out / "config.json", config)
    print(json.dumps(config), flush=True)
    rng = np.random.default_rng(args.seed)
    fixed = task_records(sample_tasks(prior, args.tasks, args.seed + 10_000)[0]) if not online else None
    best, best_epoch, history = float("inf"), 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        records = task_records(sample_tasks(prior, args.tasks, args.seed + 10_000 + epoch)[0]) if online else fixed
        loss = train_epoch(model, optimizer, records, device, args.batch_size, rng)
        scheduler.step()
        val = score(model, validation, device, args.batch_size)
        if val < best:
            best, best_epoch = val, epoch
            checkpoint(model, out, epoch, val, config)
        row = dict(epoch=epoch, synthetic_nll=loss, validation_source_macro_nmae=val,
                   best_epoch=best_epoch, elapsed_seconds=time.monotonic() - started)
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    dump_json(out / "training_summary.json", dict(best_epoch=best_epoch,
              best_validation_nmae=best, completed_epochs=len(history),
              elapsed_seconds=time.monotonic() - started,
              checkpoint_sha256=sha256(out / "best.pt"), test_evaluated=False))


def finetune(args):
    fold = Path(args.fold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    pre = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    if sha256(fold / "prior.json") != pre["training_config"]["prior_sha256"]:
        raise ValueError("Pretraining prior differs from fine-tuning fold")
    model = CreepPFN(**pre["model_config"]).to(device)
    model.load_state_dict(pre["state_dict"])
    train, ex_train = real_records(fold, "train", args.cutoff)
    val, ex_val = real_records(fold, "validation", args.cutoff)
    if not train or not val:
        raise ValueError("No eligible training or validation curves")
    by_group = {g: [r for r in train if r["group"] == g] for g in sorted(set(r["group"] for r in train))}
    groups = sorted(by_group)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    initial_val = score(model, val, device, args.batch_size)
    config = dict(arm=pre["training_config"]["arm"] + "_real_finetune", seed=args.seed,
                  fold=fold.name, model=model.config, pretrained_sha256=sha256(args.pretrained),
                  prior_sha256=sha256(fold / "prior.json"), split_sha256=sha256(fold / "split_manifest.csv"),
                  cutoff_day=args.cutoff, lr=args.lr, epochs=args.epochs, patience=args.patience,
                  batch_size=args.batch_size, train_curves=len(train), train_sources=len(groups),
                  excluded_training_curves=ex_train, validation_curves=len(val),
                  validation_sources=len(set(r["group"] for r in val)),
                  excluded_validation_curves=ex_val, gradients="real training-source future targets",
                  sampling="source-balanced with replacement; one draw per eligible training curve per epoch",
                  selection="source-macro NMAE on validation sources only", initial_validation_nmae=initial_val)
    dump_json(out / "config.json", config)
    checkpoint(model, out, 0, initial_val, config)
    best, best_epoch, history = initial_val, 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        sampled = [by_group[groups[int(rng.integers(len(groups)))]] for _ in range(len(train))]
        records = [block[int(rng.integers(len(block)))] for block in sampled]
        loss = train_epoch(model, optimizer, records, device, args.batch_size, rng)
        val_score = score(model, val, device, args.batch_size)
        if val_score < best:
            best, best_epoch = val_score, epoch
            checkpoint(model, out, epoch, val_score, config)
        row = dict(epoch=epoch, training_nll=loss, validation_source_macro_nmae=val_score,
                   best_epoch=best_epoch, elapsed_seconds=time.monotonic() - started)
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    dump_json(out / "training_summary.json", dict(best_epoch=best_epoch,
              best_validation_nmae=best, initial_validation_nmae=initial_val,
              completed_epochs=len(history), checkpoint_sha256=sha256(out / "best.pt"),
              elapsed_seconds=time.monotonic() - started, test_evaluated=False))


def evaluate(args):
    fold = Path(args.fold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ckpt["training_config"]
    if sha256(fold / "prior.json") != config["prior_sha256"]:
        raise ValueError("Evaluation prior differs from checkpoint")
    model = CreepPFN(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    records, excluded = real_records(fold, args.split, config["cutoff_day"])
    if not records:
        raise ValueError(f"No eligible {args.split} curves")
    method = config["arm"] + "_seed" + str(config["seed"])
    points = point_rows(records, predict(model, records, device), method)
    points.to_csv(out / "predictions.csv", index=False)
    dump_json(out / "metrics.json", dict(arm=config["arm"], fold=fold.name,
              split=args.split,
              seed=config["seed"], checkpoint_sha256=sha256(args.checkpoint),
              checkpoint_epoch=ckpt["epoch"], validation_nmae=ckpt["validation_score"],
              eligible_curves=len(records), excluded_curves=excluded,
              metrics=summarize(points)[method]))
    print(json.dumps({"method": method, "fold": fold.name,
                      "nmae": summarize(points, bootstrap=0)[method]["source_macro_normalized_mae"]}), flush=True)


def aggregate(args):
    root = Path(args.results)
    fold_root = Path(args.folds)
    frames, coverage = [], {}
    for pred in sorted(root.glob("fold_*/**/test/predictions.csv")):
        frame = pd.read_csv(pred)
        frames.append(frame)
        name = frame.method.iloc[0]
        coverage.setdefault(name, []).append(pred.relative_to(root).parts[0])
    if not frames:
        raise ValueError("No test predictions found")
    all_points = pd.concat(frames, ignore_index=True)
    methods = sorted(all_points.method.unique())
    excluded_by_fold = {}
    evaluable_curves = 0
    source_ids = set()
    for fold in sorted(fold_root.glob("fold_??")):
        real, excluded = real_records(fold, "test", args.cutoff)
        excluded_by_fold[fold.name] = excluded
        evaluable_curves += len(real)
        source_ids.update(r["group"] for r in real)
        if real:
            frames.append(baselines(real))
    all_points = pd.concat(frames, ignore_index=True)
    all_points.to_csv(root / "pooled_predictions.csv", index=False)
    curves = curve_metrics(all_points)
    curves.to_csv(root / "pooled_curve_metrics.csv", index=False)
    metrics = summarize(all_points)
    pairs = {}
    rng = np.random.default_rng(20260915)
    wide = curves.groupby(["group", "method"]).normalized_mae.mean().unstack("method")
    for method in methods:
        if "kelvin_prefix" not in wide or method not in wide:
            continue
        matched = wide[[method, "kelvin_prefix"]].dropna()
        delta = (matched[method] - matched.kelvin_prefix).to_numpy()
        ids = rng.integers(len(delta), size=(2000, len(delta)))
        pairs[method] = dict(matched_sources=len(matched), difference=float(delta.mean()),
                             ci95=np.quantile(delta[ids].mean(1), [.025, .975]).tolist())
    report = dict(metric="source-macro average of per-curve MAE / max(mean(abs(future increment)), 1 ue/MPa)",
                  cutoff_day=args.cutoff, outer_folds=5, eligible_imported_curves=617,
                  evaluable_outer_test_curves=evaluable_curves,
                  evaluable_outer_test_sources=len(source_ids),
                  excluded_curves_by_fold=excluded_by_fold,
                  method_fold_coverage=coverage, metrics=metrics,
                  paired_difference_vs_kelvin_prefix=pairs,
                  uncertainty="2000 percentile bootstrap replicates over source means; sources held out once")
    dump_json(root / "pooled_metrics.json", report)
    print(json.dumps({m: v["source_macro_normalized_mae"] for m, v in metrics.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--fold", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tasks", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=128)
    q = sub.add_parser("finetune")
    q.add_argument("--fold", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--pretrained", type=Path, required=True)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--epochs", type=int, default=20)
    q.add_argument("--patience", type=int, default=5)
    q.add_argument("--lr", type=float, default=1e-5)
    q.add_argument("--batch-size", type=int, default=64)
    v = sub.add_parser("evaluate")
    v.add_argument("--fold", type=Path, required=True)
    v.add_argument("--checkpoint", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--split", choices=("validation", "test"), default="test")
    a = sub.add_parser("aggregate")
    a.add_argument("--folds", type=Path, required=True)
    a.add_argument("--results", type=Path, required=True)
    for x in (p, q, v):
        x.add_argument("--cutoff", type=float, default=10.)
        x.add_argument("--device", default="auto")
        x.add_argument("--threads", type=int, default=4)
    a.add_argument("--cutoff", type=float, default=10.)
    args = parser.parse_args()
    {"fit": fit, "finetune": finetune, "evaluate": evaluate, "aggregate": aggregate}[args.command](args)


if __name__ == "__main__":
    main()

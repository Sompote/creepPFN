"""Train on synthetic tasks, select on real validation sources, evaluate once.

Model weights receive gradients from synthetic targets only. The prior itself
was calibrated on real training sources; this is a data-informed PFN prototype.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from .model import CreepPFN, predictive_quantiles, task_nll
from .pipeline import dump_json, evaluate, fit_curve, sha256, transform


def pack(records):
    nc = max(len(r["context_times"]) for r in records)
    nq = max(len(r["query_times"]) for r in records)
    n = len(records)
    result = {k: np.zeros((n, nc), np.float32) for k in ("context_times", "context_values")}
    result.update({k: np.zeros((n, nq), np.float32) for k in ("query_times", "targets")})
    result["context_mask"] = np.zeros((n, nc), bool)
    result["target_mask"] = np.zeros((n, nq), bool)
    result["features"] = np.array([r["features"] for r in records], np.float32)
    for i, r in enumerate(records):
        c, q = len(r["context_times"]), len(r["query_times"])
        for k in ("context_times", "context_values"):
            result[k][i, :c] = r[k]
        for k in ("query_times", "targets"):
            result[k][i, :q] = r[k]
        result["context_mask"][i, :c] = True
        result["target_mask"][i, :q] = True
    return result


def synthetic_records(path):
    # NpzFile decompresses a member on every lookup; materialize each array once.
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("features", "context_mask", "target_mask", "times_day", "observed_increment")}
    records = []
    for i in range(len(data["features"])):
        c, q = data["context_mask"][i], data["target_mask"][i]
        records.append(dict(context_times=data["times_day"][i, c],
                            context_values=data["observed_increment"][i, c],
                            query_times=data["times_day"][i, q],
                            targets=data["observed_increment"][i, q], features=data["features"][i]))
    return records


def real_records(prior_dir, split, cutoff=10.):
    prior_dir = Path(prior_dir)
    prior = json.loads((prior_dir / "prior.json").read_text())
    meta = pd.read_csv(prior_dir / "split_manifest.csv")
    meta = meta[meta.split == split].copy()
    raw = pd.read_csv(prior_dir / "real_observations.csv")
    raw = raw[raw.split == split]
    x = transform(meta, prior["scaler"])
    records, excluded = [], []
    for j, r in enumerate(meta.itertuples(index=False)):
        points = raw[raw.curve_id == r.curve_id].sort_values("t_day")
        t, y = points.t_day.to_numpy(float), points.increment_ue_per_MPa.to_numpy(float)
        c, q = t <= cutoff, t > cutoff
        if c.sum() < 2 or q.sum() < 1:
            excluded.append(r.curve_id)
            continue
        records.append(dict(curve_id=r.curve_id, group=r.group,
                            context_times=t[c], context_values=y[c], query_times=t[q], targets=y[q], features=x[j]))
    return records, excluded


def tensors(batch, device):
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def forward(model, batch):
    return model(**{k: batch[k] for k in
                    ("context_times", "context_values", "context_mask", "query_times", "features")})


def predict(model, records, device, batch_size=128):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            block = records[start:start + batch_size]
            batch = tensors(pack(block), device)
            output = forward(model, batch)
            median, low, high = [a.cpu().numpy() for a in predictive_quantiles(*output)]
            for i, r in enumerate(block):
                n = len(r["query_times"])
                predictions.append((median[i, :n], low[i, :n], high[i, :n]))
    return predictions


def point_rows(records, predictions, method):
    rows = []
    for r, (median, low, high) in zip(records, predictions):
        if not np.isfinite(median).all() or not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError(f"Nonfinite predictions for {r['curve_id']}")
        for j, (t, y) in enumerate(zip(r["query_times"], r["targets"])):
            rows.append(dict(curve_id=r["curve_id"], group=r["group"], method=method,
                             t_day=float(t), observed=float(y), predicted=float(median[j]),
                             lower90=float(low[j]), upper90=float(high[j])))
    return pd.DataFrame(rows)


def curve_metrics(points):
    rows = []
    for (method, group, cid), p in points.groupby(["method", "group", "curve_id"]):
        mae = float(np.mean(np.abs(p.predicted - p.observed)))
        scale = max(float(np.mean(np.abs(p.observed))), 1.)
        rows.append(dict(method=method, group=group, curve_id=cid, mae=mae,
                         normalized_mae=mae / scale,
                         coverage90=float(np.mean((p.observed >= p.lower90) & (p.observed <= p.upper90))),
                         width90=float(np.mean(p.upper90 - p.lower90)), n_points=len(p)))
    return pd.DataFrame(rows)


def summarize(points, bootstrap=2000):
    curves = curve_metrics(points)
    output = {}
    for method, frame in curves.groupby("method"):
        grouped = frame.groupby("group")[["mae", "normalized_mae", "coverage90", "width90"]].mean()
        info = dict(curves=len(frame), sources=len(grouped), observations=int(frame.n_points.sum()),
                    source_macro_mae=float(grouped.mae.mean()),
                    source_macro_normalized_mae=float(grouped.normalized_mae.mean()),
                    median_curve_normalized_mae=float(frame.normalized_mae.median()))
        if method not in ("persistence", "log_time", "kelvin_prefix"):
            info.update(source_macro_coverage90=float(grouped.coverage90.mean()),
                        source_macro_interval_width90=float(grouped.width90.mean()))
        if bootstrap:
            rng = np.random.default_rng(20260915)
            indices = rng.integers(len(grouped), size=(bootstrap, len(grouped)))
            for name in ("mae", "normalized_mae"):
                means = grouped[name].to_numpy()[indices].mean(axis=1)
                info[f"source_macro_{name}_95ci"] = np.quantile(means, [.025, .975]).tolist()
        output[method] = info
    return output


def baselines(records):
    methods = {"persistence": [], "log_time": [], "kelvin_prefix": []}
    for r in records:
        t, y, tq = r["context_times"], r["context_values"], r["query_times"]
        persistence = np.full(len(tq), y[-1])
        x = np.log1p(t) - np.log1p(t[0])
        slope = max(float(np.dot(x, y) / max(np.dot(x, x), 1e-12)), 0.)
        logarithmic = slope * (np.log1p(tq) - np.log1p(t[0]))
        params, _, _ = fit_curve("kelvin_log", t, y)
        kelvin = evaluate("kelvin_log", params, tq, t[0])
        for method, values in (("persistence", persistence), ("log_time", logarithmic), ("kelvin_prefix", kelvin)):
            methods[method].append((values, values, values))
    return pd.concat([point_rows(records, values, name) for name, values in methods.items()], ignore_index=True)


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    records = synthetic_records(args.tasks)
    validation, excluded = real_records(args.prior_dir, "validation", args.cutoff)
    if not validation:
        raise ValueError("No eligible real validation curves")
    model = CreepPFN(args.width, args.layers, 4, .05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    rng = np.random.default_rng(args.seed)
    parameter_count = sum(p.numel() for p in model.parameters())
    config = dict(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  model=model.config, parameters=parameter_count, device=str(device), threads=args.threads,
                  synthetic_tasks=len(records), validation_curves=len(validation),
                  validation_sources=len(set(r["group"] for r in validation)),
                  excluded_validation_curves=excluded, cutoff_day=args.cutoff,
                  selection_metric="source_macro_normalized_mae on real validation",
                  normalization="curve MAE / max(mean(abs(future measured increment)), 1 ue/MPa)",
                  gradients="synthetic noisy targets only", torch_version=str(torch.__version__),
                  numpy_version=np.__version__, tasks_sha256=sha256(args.tasks),
                  prior_sha256=sha256(Path(args.prior_dir) / "prior.json"),
                  code_sha256={p: sha256(Path(__file__).parent / p) for p in ("model.py", "train.py")})
    dump_json(out / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)
    baseline_points = baselines(validation)
    dump_json(out / "validation_baselines.json", summarize(baseline_points))
    best, best_epoch, history = float("inf"), 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(records))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = tensors(pack([records[i] for i in order[start:start + args.batch_size]]), device)
            optimizer.zero_grad(set_to_none=True)
            mu, sigma, scale = forward(model, batch)
            loss = task_nll(mu, sigma, scale, batch["targets"], batch["target_mask"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append((float(loss.detach().cpu()), len(batch["features"])))
        scheduler.step()
        predictions = predict(model, validation, device, args.batch_size)
        points = point_rows(validation, predictions, "creep_pfn")
        metrics = summarize(points, bootstrap=0)["creep_pfn"]
        score = metrics["source_macro_normalized_mae"]
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            model_config=model.config, epoch=epoch, validation_score=score,
                            training_config=config), out / "best.pt")
            points.to_csv(out / "validation_predictions.csv", index=False)
        row = dict(epoch=epoch, training_nll=float(np.average([v for v, n in losses], weights=[n for v, n in losses])),
                   validation_nmae=score, validation_mae=metrics["source_macro_mae"],
                   validation_coverage90=metrics["source_macro_coverage90"],
                   elapsed_seconds=time.monotonic() - started, best_epoch=best_epoch)
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement", flush=True)
            break
    dump_json(out / "training_summary.json", dict(best_epoch=best_epoch, best_validation_nmae=best,
              completed_epochs=len(history), elapsed_seconds=time.monotonic() - started,
              test_evaluated=False, checkpoint_sha256=sha256(out / "best.pt")))


def evaluate_command(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["training_config"]
    if sha256(Path(args.prior_dir) / "prior.json") != config["prior_sha256"]:
        raise ValueError("Evaluation prior does not match training prior")
    model = CreepPFN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    records, excluded = real_records(args.prior_dir, args.split, config["cutoff_day"])
    predictions = predict(model, records, device)
    points = pd.concat([point_rows(records, predictions, "creep_pfn"), baselines(records)], ignore_index=True)
    points.to_csv(out / "predictions.csv", index=False)
    curves = curve_metrics(points)
    curves.to_csv(out / "per_curve.csv", index=False)
    metrics = summarize(points)
    report = dict(split=args.split, cutoff_day=config["cutoff_day"], checkpoint_epoch=checkpoint["epoch"],
                  checkpoint_sha256=sha256(args.checkpoint), excluded_curves=excluded, metrics=metrics,
                  interval="central 90% marginal intervals; no real-data interval recalibration",
                  uncertainty="95% percentile bootstrap over source means, 2000 replicates; one model seed",
                  target="compliance increment relative to first observation; raw future measurements",
                  model_selection="real validation sources only; no test-based tuning")
    # Paired source bootstrap compares methods on the same held-out groups.
    grouped = curves.groupby(["method", "group"]).normalized_mae.mean().unstack(0)
    rng = np.random.default_rng(20260915)
    indices = rng.integers(len(grouped), size=(2000, len(grouped)))
    report["paired_nmae_difference_pfn_minus_baseline"] = {}
    for method in ("persistence", "log_time", "kelvin_prefix"):
        delta = (grouped.creep_pfn - grouped[method]).to_numpy()
        report["paired_nmae_difference_pfn_minus_baseline"][method] = dict(
            difference=float(delta.mean()), ci95=np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist())
    dump_json(out / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--tasks", type=Path, required=True)
    fit.add_argument("--prior-dir", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=40)
    fit.add_argument("--patience", type=int, default=10)
    fit.add_argument("--batch-size", type=int, default=128)
    fit.add_argument("--width", type=int, default=64)
    fit.add_argument("--layers", type=int, default=2)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--cutoff", type=float, default=10.)
    fit.set_defaults(func=train)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--prior-dir", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.set_defaults(func=evaluate_command)
    for command in (fit, evaluation):
        command.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "fit" and (args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.width % 4):
        parser.error("Positive training settings and width divisible by four are required")
    args.func(args)


if __name__ == "__main__":
    main()

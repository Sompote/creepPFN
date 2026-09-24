"""Command-line interface for prediction, verification, training, and Gradio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .deployment import forecast_ensemble, forecast_figure, repository_root, validate_bundle


def _predict(args):
    context = pd.read_csv(args.context)
    required = {"t_day", "compliance"}
    if not required.issubset(context.columns):
        raise ValueError("Context CSV must contain t_day and compliance columns.")
    torch.set_num_threads(args.threads)
    result = forecast_ensemble(
        context.t_day,
        context.compliance,
        args.query_days,
        args.rho,
        args.fc,
        args.E28,
        stress_ratio=args.stress_ratio,
        fold=args.fold,
        device=args.device,
        root=args.root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out}. Add --overwrite to replace it.")
    result.to_csv(args.out, index=False)
    if args.plot:
        figure = forecast_figure(context.t_day.to_numpy(), context.compliance.to_numpy(), result)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=180, bbox_inches="tight")
    print(result.to_string(index=False))
    print(f"\nWrote {args.out}")


def _serve(args):
    from .gradio_app import create_demo

    create_demo(root=args.root).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


def _verify(args):
    print(json.dumps(validate_bundle(args.root), indent=2))


def _train(args):
    """Reproduce one fold/seed run: pretraining on the hierarchical prior, then fine-tuning."""
    from . import architecture_study as study
    from .bayes.sampler import sample_tasks

    study.sample_tasks = sample_tasks          # synthetic tasks from the hierarchical Bayesian prior
    bundle = repository_root(args.root)
    fold_name = args.fold if args.fold.startswith("fold_") else f"fold_{int(args.fold):02d}"
    choice = json.loads((bundle / "data/training_settings.json").read_text())
    if fold_name not in choice["fold_settings"]:
        raise ValueError(f"Unknown fold: {fold_name}")
    selected = choice["fold_settings"][fold_name]
    study.fit_run(SimpleNamespace(
        fold=bundle / "data/folds" / fold_name,
        out=args.out,
        label="hier_full",
        model_config=json.dumps(selected["model_config"]),
        pre_lr=selected["pre_lr"],
        fine_lr=selected["fine_lr"],
        seed=args.seed,
        tasks=args.tasks,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        threads=args.threads,
    ))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="creep-pfn", description=__doc__)
    root.add_argument("--root", type=Path, help="Supplementary repository root.")
    sub = root.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Forecast from a context CSV.")
    predict.add_argument("--context", type=Path, required=True,
                         help="CSV with t_day and compliance (microstrain/MPa).")
    predict.add_argument("--rho", type=float, required=True, help="Density in kg/m^3.")
    predict.add_argument("--fc", type=float, required=True, help="Compressive strength in MPa.")
    predict.add_argument("--E28", type=float, default=float("nan"),
                         help="28-day modulus in MPa; omit when unavailable.")
    predict.add_argument("--stress-ratio", type=float, default=float("nan"),
                         help="Applied stress / compressive strength at loading; omit when unavailable.")
    predict.add_argument("--query-days", type=float, nargs="+", required=True)
    predict.add_argument("--fold", default="all",
                         choices=("all", "fold_01", "fold_02", "fold_03", "fold_04", "fold_05"))
    predict.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    predict.add_argument("--threads", type=int, default=4)
    predict.add_argument("--out", type=Path, default=Path("predictions.csv"))
    predict.add_argument("--plot", type=Path)
    predict.add_argument("--overwrite", action="store_true")
    predict.set_defaults(func=_predict)

    app = sub.add_parser("app", help="Launch the Gradio interface.")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=7860)
    app.add_argument("--share", action="store_true")
    app.add_argument("--inbrowser", action="store_true")
    app.set_defaults(func=_serve)

    verify = sub.add_parser("verify", help="Validate packaged priors and checkpoints.")
    verify.set_defaults(func=_verify)

    train = sub.add_parser("train", help="Reproduce one fold/seed run on the hierarchical prior.")
    train.add_argument("--fold", default="fold_01",
                       choices=("fold_01", "fold_02", "fold_03", "fold_04", "fold_05"))
    train.add_argument("--seed", type=int, default=45)
    train.add_argument("--tasks", type=int, default=5000, help="Fresh synthetic tasks per epoch.")
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--patience", type=int, default=6)
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    train.add_argument("--threads", type=int, default=8)
    train.add_argument("--out", type=Path, required=True)
    train.set_defaults(func=_train)
    return root


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

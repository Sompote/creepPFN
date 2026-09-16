"""Resumable driver for the prespecified five-fold comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ARMS = ("fixed_small", "online_small", "online_medium", "online_large")
FINETUNE = ("online_medium", "online_large")


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        print(log.read_text()[-4000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"Command failed ({proc.returncode}): {command}; see {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--phase", choices=("fit", "repeat", "evaluate", "evaluate_repeat"), required=True)
    p.add_argument("--tasks", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    folds = sorted(args.folds.glob("fold_??"))
    if len(folds) != 5:
        raise ValueError("Expected exactly five folds")
    selected = None
    if args.phase in ("repeat", "evaluate_repeat"):
        selected = json.loads((args.results / "selection_per_fold.json").read_text())
    work = []
    for fold in folds:
        arms = ARMS if selected is None else (selected[fold.name]["winner"]["arm"],)
        for arm in arms:
            directory = args.results / fold.name / arm / f"seed{args.seed}"
            if args.phase in ("fit", "repeat"):
                work.append((directory, [sys.executable, "-m", "creep_prior.cv_experiment", "fit",
                         "--fold", str(fold), "--out", str(directory), "--arm", arm,
                         "--seed", str(args.seed), "--tasks", str(args.tasks),
                         "--epochs", str(args.epochs), "--device", args.device,
                         "--threads", str(args.threads)], "training_summary.json"))
            selected_tune = selected is not None and selected[fold.name]["winner"]["real_finetune"]
            if arm in FINETUNE and (selected is None or selected_tune):
                tuned = directory / "real_finetune"
                if args.phase in ("fit", "repeat"):
                    work.append((tuned, [sys.executable, "-m", "creep_prior.cv_experiment", "finetune",
                                 "--fold", str(fold), "--out", str(tuned),
                                 "--pretrained", str(directory / "best.pt"),
                                 "--seed", str(args.seed), "--device", args.device,
                                 "--threads", str(args.threads)], "training_summary.json"))
            if args.phase in ("evaluate", "evaluate_repeat"):
                targets = [(directory, directory / "best.pt")]
                if arm in FINETUNE and (selected is None or selected_tune):
                    targets.append((directory / "real_finetune", directory / "real_finetune" / "best.pt"))
                if args.phase == "evaluate_repeat":
                    targets = [targets[-1]] if selected_tune else targets[:1]
                for base, checkpoint in targets:
                    test = base / "test"
                    work.append((test, [sys.executable, "-m", "creep_prior.cv_experiment", "evaluate",
                                 "--fold", str(fold), "--checkpoint", str(checkpoint),
                                 "--out", str(test), "--device", args.device,
                                 "--threads", str(args.threads)], "metrics.json"))
    started = time.monotonic()
    for number, (directory, command, sentinel) in enumerate(work, 1):
        if (directory / sentinel).exists():
            print(json.dumps({"job": number, "total": len(work), "status": "already complete",
                              "path": str(directory)}), flush=True)
            continue
        if directory.exists():
            raise FileExistsError(f"Incomplete output exists: {directory}; inspect before retrying")
        print(json.dumps({"job": number, "total": len(work), "status": "starting",
                          "path": str(directory), "elapsed_seconds": time.monotonic() - started}), flush=True)
        run(command, args.results / "logs" / f"{args.phase}_{fold_name(directory)}.log")
        print(json.dumps({"job": number, "total": len(work), "status": "completed",
                          "path": str(directory), "elapsed_seconds": time.monotonic() - started}), flush=True)
def fold_name(directory):
    return "_".join(directory.parts[-4:]).replace("/", "_")


if __name__ == "__main__":
    main()

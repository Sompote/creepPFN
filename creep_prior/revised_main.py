"""Retrain the validation-favored architecture with fold-selected settings.

The original NU outer-test sources have been examined in earlier studies.
Consequently, this revised fit is a post-selection reassessment, even though
its architecture rule and fold hyperparameters use saved validation scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .architecture_study import ABLATIONS, ensemble_points, run_fit, worker
from .pipeline import dump_json, sha256, transform
from .train import baselines, curve_metrics, summarize

SEEDS = (45, 46, 47)
ARMS = ("full",) + tuple(ABLATIONS)
ROOT = Path(__file__).resolve().parents[1]


def folds_at(path: Path):
    folds = sorted(path.glob("fold_??"))
    if len(folds) != 5:
        raise ValueError("Expected five source-disjoint NU folds")
    return folds


def choose(args):
    study, results = Path(args.architecture_study), Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    values = {arm: [] for arm in ARMS}
    settings = {}
    for fold in folds_at(Path(args.folds)):
        selected = json.loads((study / fold.name / "tuning_selection.json").read_text())["winner"]
        settings[fold.name] = dict(model_config=selected["model_config"],
                                   pre_lr=selected["pre_lr"], fine_lr=selected["fine_lr"],
                                   hyperparameter_candidate=selected["name"])
        for arm in ARMS:
            fit_label = (f"tune_{selected['name']}" if arm == "full"
                         else f"ablate_{arm}")
            path = study / fold.name / fit_label / "seed42/summary.json"
            values[arm].append(json.loads(path.read_text())["best_validation_nmae"])
    ranking = sorted((dict(arm=arm, mean_of_five_fold_validation_nmae=float(np.mean(scores)),
                           fold_validation_nmae=scores) for arm, scores in values.items()),
                     key=lambda item: (item["mean_of_five_fold_validation_nmae"], item["arm"]))
    winner = ranking[0]["arm"]
    if winner != "full":
        raise ValueError(f"Saved validation rule selected {winner}, not the planned full architecture")
    dump_json(results / "architecture_choice.json", dict(
        architecture=winner, ranking=ranking, fold_settings=settings, seeds=SEEDS,
        rule="lowest equal-fold mean of saved seed-42 source-macro validation NMAE",
        hyperparameters="previous fold-specific validation winner among 11 one-factor candidates",
        qualification="retrospective architecture choice after prior NU outer-test exposure; reassessment is exploratory",
        architecture_study_sha256=sha256(study / "results.json"),
        revised_code_sha256=sha256(__file__)))
    print(json.dumps(dict(chosen_architecture=winner, validation_ranking=ranking)), flush=True)


def train(args):
    results = Path(args.results)
    choice_path = results / "architecture_choice.json"
    choice = json.loads(choice_path.read_text())
    if choice["architecture"] != "full":
        raise ValueError("This runner expects the saved full architecture choice")
    for fold in folds_at(Path(args.folds)):
        setting = choice["fold_settings"][fold.name]
        for seed in SEEDS:
            run_fit(results, fold, "revised_main", setting["model_config"],
                    setting["pre_lr"], setting["fine_lr"], seed, args)
        print(json.dumps(dict(fold=fold.name, revised_fit_complete=True)), flush=True)
    dump_json(results / "fit_complete.json", dict(
        fits=15, folds=5, seeds=SEEDS, architecture="full",
        choice_sha256=sha256(choice_path),
        outer_test_access="none during fit; earlier experiments had already exposed these sources"))


def evaluate(args):
    results = Path(args.results)
    if not (results / "fit_complete.json").exists():
        raise FileNotFoundError("All revised fits must finish before outer-test evaluation")
    for fold in folds_at(Path(args.folds)):
        for seed in SEEDS:
            checkpoint = results / fold.name / "revised_main" / f"seed{seed}" / "finetune/best.pt"
            out = results / fold.name / "test" / f"seed{seed}"
            if (out / "description.json").exists():
                continue
            if out.exists():
                raise FileExistsError(f"Incomplete test evaluation: {out}")
            cmd = [sys.executable, "-m", "creep_prior.architecture_study", "evaluate-run",
                   "--fold", str(fold), "--checkpoint", str(checkpoint),
                   "--out", str(out), "--label", "revised_main", "--seed", str(seed),
                   "--device", args.device, "--threads", str(args.threads)]
            worker(cmd, results / "logs" / f"test_{fold.name}_seed{seed}.log")
            print(json.dumps(dict(phase="outer_test", fold=fold.name, seed=seed)), flush=True)
    dump_json(results / "evaluation_complete.json", dict(
        member_evaluations=15, fit_complete_sha256=sha256(results / "fit_complete.json")))


def paired_effect(new_curves: pd.DataFrame, old_points: pd.DataFrame, old_method: str):
    old = curve_metrics(old_points[old_points.method == old_method])
    cols = ["group", "curve_id"]
    left = new_curves[cols + ["normalized_mae"]].rename(columns={"normalized_mae": "new"})
    right = old[cols + ["normalized_mae"]].rename(columns={"normalized_mae": "old"})
    merged = left.merge(right, on=cols, validate="one_to_one")
    if len(merged) != 610 or merged.group.nunique() != 65:
        raise ValueError("Revised and reference methods do not share the evaluation cohort")
    sources = merged.groupby("group")[["new", "old"]].mean()
    delta = (sources.new - sources.old).to_numpy()
    ids = np.random.default_rng(20260915).integers(65, size=(2000, 65))
    return dict(reference=old_method, new_minus_reference_pctpt=float(delta.mean() * 100),
                ci95_pctpt=(np.quantile(delta[ids].mean(axis=1), [.025, .975]) * 100).tolist(),
                matched_sources=65)


def assert_matching_observations(new_points: pd.DataFrame, old_points: pd.DataFrame,
                                 old_method: str):
    columns = ["curve_id", "group", "t_day", "observed"]
    sort = ["curve_id", "t_day"]
    left = new_points[columns].sort_values(sort).reset_index(drop=True)
    right = old_points.loc[old_points.method == old_method, columns].sort_values(sort).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(f"Revised and {old_method} targets or schedules differ")


def analyze(args):
    results = Path(args.results)
    if not (results / "evaluation_complete.json").exists():
        raise FileNotFoundError("Member evaluations are incomplete")
    frames = []
    for fold in folds_at(Path(args.folds)):
        members = [pd.read_csv(results / fold.name / "test" / f"seed{seed}" / "predictions.csv")
                   for seed in SEEDS]
        frames.append(ensemble_points(members, fold, "revised_main"))
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(results / "pooled_predictions.csv", index=False)
    curves = curve_metrics(pooled)
    curves.to_csv(results / "pooled_curve_metrics.csv", index=False)
    metrics = summarize(pooled)["revised_main"]
    if (metrics["curves"], metrics["sources"], metrics["observations"]) != (610, 65, 5646):
        raise ValueError("Unexpected revised NU evaluation cohort")
    historical = pd.read_csv(args.historical_predictions)
    tuned = pd.read_csv(args.architecture_predictions)
    assert_matching_observations(pooled, historical, "selected_ensemble_3seed")
    assert_matching_observations(pooled, tuned, "full")
    paired = dict(
        historical=paired_effect(curves, historical, "selected_ensemble_3seed"),
        tuned_full=paired_effect(curves, tuned, "full"))
    fold_metrics = {fold: summarize(points, bootstrap=0)["revised_main"]
                    for fold, points in pooled.groupby("fold")}
    dump_json(results / "results.json", dict(
        protocol="post-selection retraining at saved fold-validation hyperparameters; five NU source-disjoint folds",
        architecture="full width-256, four-layer, four-head CreepPFN", seeds=SEEDS,
        cohort=dict(curves=610, sources=65, future_readings=5646),
        choice_sha256=sha256(results / "architecture_choice.json"),
        selection_limit="prior NU outer-test outcomes had been examined; this reassessment is exploratory, not a new independent confirmation",
        metrics=metrics, fold_metrics=fold_metrics, paired=paired,
        uncertainty="2000 source bootstrap draws; fixed seeds and settings; selection uncertainty omitted"))
    print(json.dumps(dict(revised_source_macro_nmae=metrics["source_macro_normalized_mae"],
                          paired=paired)), flush=True)


def external(args):
    from .external_sources import (descriptive_metrics,
                                   ensemble_points as external_ensemble,
                                   load_external, member_predictions)
    results, out = Path(args.results), Path(args.out)
    if not (results / "results.json").exists():
        raise FileNotFoundError("Revised NU analysis must complete before external forecasting")
    out.mkdir(parents=True, exist_ok=True)
    meta, observations = load_external(Path(args.workbook), Path(args.metadata), args.modulus_mode)
    meta.to_csv(out / "cohort_audit.csv", index=False)
    context = [dict(curve_id=r["curve_id"], group=r["group"],
                    t_day=float(t), increment_ue_per_MPa=float(y))
               for r in observations for t, y in zip(r["context_times"], r["context_values"])]
    pd.DataFrame(context).to_csv(out / "context_points.csv", index=False)
    frames, fold_metrics, checkpoints = [], {}, {}
    base = baselines(observations)
    for fold in folds_at(Path(args.folds)):
        prior = json.loads((fold / "prior.json").read_text())
        split = pd.read_csv(fold / "split_manifest.csv")
        if any(str(x).startswith("PA") for x in prior["training_curve_ids"]) or \
                any(str(x).startswith("PA") for x in split.curve_id):
            raise ValueError("External records entered revised training data")
        features = transform(meta, prior["scaler"])
        records = [dict(r, features=features[i]) for i, r in enumerate(observations)]
        members = []
        checkpoints[fold.name] = []
        for seed in SEEDS:
            path = results / fold.name / "revised_main" / f"seed{seed}" / "finetune/best.pt"
            member, weights = member_predictions(records, path, args.threads)
            if weights["training_config"]["prior_sha256"] != sha256(fold / "prior.json"):
                raise ValueError("Revised member prior/fold mismatch")
            members.append(member)
            checkpoints[fold.name].append(dict(seed=seed, sha256=sha256(path), path=str(path)))
        ensemble = external_ensemble(records, members, fold.name)
        all_points = pd.concat([ensemble, base.assign(fold=fold.name)], ignore_index=True)
        fold_metrics[fold.name] = descriptive_metrics(all_points)
        frames.append(all_points)
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(out / "predictions_by_fold.csv", index=False)
    origins = ("KMUTT", "other_literature", "pooled")
    mean_scores = {origin: float(np.mean([
        fold_metrics[fold]["external_ensemble"][origin]["curve_macro_nmae_pct"]
        for fold in fold_metrics])) for origin in origins}
    ranges = {origin: [float(min(
        fold_metrics[fold]["external_ensemble"][origin]["curve_macro_nmae_pct"]
        for fold in fold_metrics)), float(max(
        fold_metrics[fold]["external_ensemble"][origin]["curve_macro_nmae_pct"]
        for fold in fold_metrics))] for origin in origins}
    dump_json(out / "results.json", dict(
        protocol="five frozen revised NU ensembles tested on the same 66 supplementary curves",
        modulus_mode=args.modulus_mode,
        target="spreadsheet strain / nominal 0.4*fc, minus first-reading response",
        selection_limit="external cohorts previously examined with historical model; transfer reassessment is descriptive",
        literature_provenance="25 curves lack per-publication identifiers and measured loading stress",
        input_hashes=dict(workbook=sha256(args.workbook), metadata=sha256(args.metadata)),
        checkpoint_hashes=checkpoints, fold_metrics=fold_metrics,
        mean_fold_curve_macro_nmae_pct=mean_scores, fold_range_curve_macro_nmae_pct=ranges))
    print(json.dumps(dict(modulus_mode=args.modulus_mode,
                          mean_fold_curve_macro_nmae_pct=mean_scores)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, default=ROOT / "artifacts/cv5_priors")
    parser.add_argument("--architecture-study", type=Path,
                        default=ROOT / "artifacts/architecture_study")
    parser.add_argument("--historical-predictions", type=Path,
                        default=ROOT / "artifacts/cv5_results/pooled_predictions.csv")
    parser.add_argument("--architecture-predictions", type=Path,
                        default=ROOT / "artifacts/architecture_study/pooled_predictions.csv")
    parser.add_argument("--results", type=Path, default=ROOT / "artifacts/revised_main")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--modulus-mode", choices=("recorded_proxy", "missing"),
                        default="recorded_proxy")
    parser.add_argument("--out", type=Path)
    parser.add_argument("phase", choices=("choose", "train", "evaluate", "analyze", "external"))
    args = parser.parse_args()
    if args.phase == "external" and (args.workbook is None or args.metadata is None or args.out is None):
        parser.error("external phase requires --workbook, --metadata and --out")
    {"choose": choose, "train": train, "evaluate": evaluate,
     "analyze": analyze, "external": external}[args.phase](args)


if __name__ == "__main__":
    main()

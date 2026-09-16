"""Evaluate frozen NU-trained CreepPFN ensembles on supplementary raw curves.

The spreadsheet does not identify which publication supplied each literature
curve or the age at which its elastic modulus was measured. The recorded modulus
can enter the E28 slot as an explicitly labelled proxy, or be treated as missing
for sensitivity analysis. No external curve is used for prior fitting, gradient
updates, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import torch
from scipy.special import ndtr

from .model import CreepPFN
from .pipeline import dump_json, sha256, transform
from .train import baselines, curve_metrics, point_rows, predict


KSC_TO_MPA = 0.0980665
Z90 = 1.6448536269514722


def load_external(workbook: Path, metadata: Path, modulus_mode: str = "recorded_proxy"):
    """Read original spreadsheet values; never use stored daily curve fits."""
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    sheet_y, sheet_properties = book["Sheet1"], book["Sheet2"]
    y_rows = list(sheet_y.values)
    properties = list(sheet_properties.values)
    days = np.asarray(y_rows[0], float)
    if properties[0][:3] != ("Density", "fc", "E"):
        raise ValueError("Unexpected property-column order")
    if len(y_rows) - 1 != 66 or len(properties) - 1 != 66:
        raise ValueError("Expected 66 aligned spreadsheet curves")
    metadata_frame = pd.read_csv(metadata).set_index("curve_id", verify_integrity=True)
    records, rows = [], []
    for i, (reading_row, descriptor_row) in enumerate(zip(y_rows[1:], properties[1:])):
        cid = f"PA{i:03d}"
        if cid not in metadata_frame.index:
            raise ValueError(f"Missing metadata row: {cid}")
        stored = metadata_frame.loc[cid]
        group = "PAPER_lab" if i < 41 else "PAPER_lit"
        if stored.group != group:
            raise ValueError(f"Unexpected group for {cid}: {stored.group}")
        rho = float(descriptor_row[0])
        fc = float(descriptor_row[1]) * KSC_TO_MPA
        e_recorded = float(descriptor_row[2]) * KSC_TO_MPA
        if not (np.isfinite(rho) and rho > 0 and np.isfinite(fc) and fc > 0 and
                np.isfinite(e_recorded) and e_recorded > 0):
            raise ValueError(f"Invalid descriptors for {cid}")
        if not (np.isclose(rho, stored.rho) and np.isclose(fc, stored.fc) and
                np.isclose(e_recorded, stored.E)):
            raise ValueError(f"Metadata and workbook properties differ for {cid}")
        strain = np.array([np.nan if v is None else float(v) for v in reading_row])
        mask = np.isfinite(strain) & np.isfinite(days) & (days >= 0) & (days <= 160)
        t = days[mask]
        # Spreadsheet strain is microstrain; 0.4*fc is the nominal load stress.
        # Subtracting the first reading gives the same increment target form as NU.
        y = strain[mask] / (0.4 * fc)
        y = y - y[0]
        if len(t) < 4 or t[0] > 10 or t[-1] < 28 or np.any(np.diff(t) <= 0):
            raise ValueError(f"Unexpected original observation schedule for {cid}")
        context, query = t <= 10, t > 10
        if context.sum() < 2 or query.sum() < 1:
            raise ValueError(f"Curve {cid} is not day-10 eligible")
        rows.append(dict(curve_id=cid, group=group, rho=rho, fc=fc,
                         E_recorded=e_recorded,
                         E28=e_recorded if modulus_mode == "recorded_proxy" else np.nan,
                         anchor_day=t[0], n_obs=len(t),
                         last_day=t[-1], context_n=int(context.sum()),
                         future_n=int(query.sum())))
        records.append(dict(curve_id=cid, group=group, context_times=t[context],
                            context_values=y[context], query_times=t[query],
                            targets=y[query]))
    return pd.DataFrame(rows), records


def member_predictions(records, checkpoint: Path, threads: int):
    torch.set_num_threads(threads)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CreepPFN(**weights["model_config"])
    model.load_state_dict(weights["state_dict"])
    return predict(model, records, torch.device("cpu")), weights


def ensemble_points(records, members, fold):
    base = point_rows(records, members[0], "external_ensemble")
    for other in members[1:]:
        check = point_rows(records, other, "external_ensemble")
        if not base[["curve_id", "group", "t_day", "observed"]].equals(
                check[["curve_id", "group", "t_day", "observed"]]):
            raise ValueError("External member predictions are misaligned")
    frames = [point_rows(records, x, "external_ensemble") for x in members]
    scale_map = {r["curve_id"]: max(float(np.max(np.abs(r["context_values"]))), 1.)
                 for r in records}
    scale = np.array([scale_map[c] for c in base.curve_id], float)
    base["predicted"] = np.mean([f.predicted.to_numpy() for f in frames], axis=0)
    mu = np.stack([np.arcsinh(f.predicted.to_numpy() / scale) for f in frames])
    low = np.stack([np.arcsinh(f.lower90.to_numpy() / scale) for f in frames])
    high = np.stack([np.arcsinh(f.upper90.to_numpy() / scale) for f in frames])
    sigma = np.maximum((high - low) / (2 * Z90), 1e-6)

    def quantile(probability):
        lower = np.min(mu - 8 * sigma, axis=0)
        upper = np.max(mu + 8 * sigma, axis=0)
        for _ in range(48):
            middle = (lower + upper) / 2
            cdf = ndtr((middle[None, :] - mu) / sigma).mean(axis=0)
            lower = np.where(cdf < probability, middle, lower)
            upper = np.where(cdf >= probability, middle, upper)
        return np.sinh((lower + upper) / 2) * scale

    base["lower90"] = quantile(.05)
    base["upper90"] = quantile(.95)
    base["fold"] = fold
    return base


def descriptive_metrics(points):
    curves = curve_metrics(points)
    output = {}
    for method, by_method in curves.groupby("method"):
        output[method] = {}
        for origin, frame in (("pooled", by_method),
                              ("KMUTT", by_method[by_method.group == "PAPER_lab"]),
                              ("other_literature", by_method[by_method.group == "PAPER_lit"])):
            info = dict(curves=len(frame), future_readings=int(frame.n_points.sum()),
                        curve_macro_nmae_pct=float(frame.normalized_mae.mean() * 100),
                        median_curve_nmae_pct=float(frame.normalized_mae.median() * 100),
                        curve_macro_mae=float(frame.mae.mean()))
            if method == "external_ensemble":
                info.update(curve_macro_coverage90=float(frame.coverage90.mean()),
                            curve_macro_width90=float(frame.width90.mean()))
            output[method][origin] = info
    return output


def evaluate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    meta, observations = load_external(args.workbook, args.metadata, args.modulus_mode)
    meta.to_csv(args.out / "cohort_audit.csv", index=False)
    context_rows = [dict(curve_id=r["curve_id"], group=r["group"],
                         t_day=float(t), increment_ue_per_MPa=float(y))
                    for r in observations
                    for t, y in zip(r["context_times"], r["context_values"])]
    pd.DataFrame(context_rows).to_csv(args.out / "context_points.csv", index=False)
    selection = json.loads((args.cv_results / "selection_per_fold.json").read_text())
    base = baselines(observations)
    frames, metrics, checkpoints = [], {}, {}
    for fold, chosen in sorted(selection.items()):
        prior = json.loads((args.cv_priors / fold / "prior.json").read_text())
        split = pd.read_csv(args.cv_priors / fold / "split_manifest.csv")
        if any(str(cid).startswith("PA") for cid in prior["training_curve_ids"]):
            raise ValueError(f"External curve entered the synthetic prior in {fold}")
        if any(str(cid).startswith("PA") for cid in split.curve_id):
            raise ValueError(f"External curve entered the NU training/validation/test split in {fold}")
        features = transform(meta, prior["scaler"])
        records = [dict(r, features=features[i]) for i, r in enumerate(observations)]
        member_rows = []
        checkpoints[fold] = []
        for seed in (42, 43, 44):
            path = args.cv_results / fold / chosen["winner"]["arm"] / f"seed{seed}"
            if chosen["winner"]["real_finetune"]:
                path /= "real_finetune"
            path /= "best.pt"
            member, weights = member_predictions(records, path, args.threads)
            if weights["training_config"]["prior_sha256"] != sha256(args.cv_priors / fold / "prior.json"):
                raise ValueError(f"Checkpoint/prior mismatch in {fold}")
            member_rows.append(member)
            checkpoints[fold].append(dict(path=str(path), sha256=sha256(path), seed=seed))
        ensemble = ensemble_points(records, member_rows, fold)
        all_points = pd.concat([ensemble, base.assign(fold=fold)], ignore_index=True)
        metrics[fold] = descriptive_metrics(all_points)
        frames.append(all_points)
        print(json.dumps({"fold": fold,
                          "KMUTT_nmae_pct": metrics[fold]["external_ensemble"]["KMUTT"]["curve_macro_nmae_pct"],
                          "other_nmae_pct": metrics[fold]["external_ensemble"]["other_literature"]["curve_macro_nmae_pct"]}), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(args.out / "predictions_by_fold.csv", index=False)
    dump_json(args.out / "results.json", dict(
        protocol="five frozen, NU-trained three-seed ensembles each evaluated on the same 66 supplementary curves",
        model_selection="existing NU development validation; external targets never used for fitting or selection",
        target="raw spreadsheet strain divided by nominal 0.4*fc, then increment from first reading",
        modulus=("recorded spreadsheet E supplied as E28-slot proxy; measurement age undocumented"
                 if args.modulus_mode == "recorded_proxy" else
                 "E28 treated as missing; spreadsheet E measurement age undocumented"),
        modulus_mode=args.modulus_mode,
        literature_provenance="25 curves have one generic label and no per-article identifiers",
        uncertainty="fold variability is descriptive; the 41 KMUTT curves and 25 unresolved literature curves do not define independent source replications",
        input_hashes={"workbook": sha256(args.workbook), "metadata": sha256(args.metadata),
                      "external_evaluation_code": sha256(__file__)},
        checkpoint_hashes=checkpoints, fold_metrics=metrics))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cv-priors", type=Path, default=root / "artifacts/cv5_priors")
    parser.add_argument("--cv-results", type=Path, default=root / "artifacts/cv5_results")
    parser.add_argument("--out", type=Path, default=root / "artifacts/external_sources")
    parser.add_argument("--modulus-mode", choices=("recorded_proxy", "missing"),
                        default="recorded_proxy")
    parser.add_argument("--threads", type=int, default=4)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

"""Build source-disjoint outer folds and training-only synthetic priors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .pipeline import (
    FAMILIES, FEATURES, dump_json, export_real, fit_family, fit_scaler,
    load_original, sha256,
)


def build(data_dir: Path, out: Path, seed: int = 20260915, folds: int = 5):
    out.mkdir(parents=True, exist_ok=False)
    meta, observations, audit = load_original(data_dir)
    meta = meta.sort_values("curve_id").reset_index(drop=True)
    splitter = GroupKFold(n_splits=folds)
    summary = dict(seed=seed, outer_folds=folds, eligible_curves=len(meta),
                   eligible_groups=meta.group.nunique(), input_hashes=audit["input_hashes"], folds=[])
    all_outer = []
    for fold, (dev_idx, test_idx) in enumerate(splitter.split(meta, groups=meta.group), 1):
        fold_dir = out / f"fold_{fold:02d}"
        fold_dir.mkdir()
        dev_groups = np.array(sorted(meta.iloc[dev_idx].group.unique()))
        rng = np.random.default_rng(seed + fold)
        rng.shuffle(dev_groups)
        val_groups = set(dev_groups[:max(1, round(.2 * len(dev_groups)))])
        test_groups = set(meta.iloc[test_idx].group)
        split = np.where(meta.group.isin(test_groups), "test",
                         np.where(meta.group.isin(val_groups), "validation", "train"))
        frame = meta.assign(split=split)
        train = frame[frame.split == "train"].reset_index(drop=True)
        scaler = fit_scaler(train)
        prior = dict(schema_version=1, horizon_day=160., seed=seed + fold,
                     feature_names=FEATURES, scaler=scaler, families={},
                     training_curve_ids=train.curve_id.tolist(),
                     training_groups=sorted(train.group.unique()),
                     target="compliance increment relative to first observation",
                     property_E="independently recorded 28-day modulus; missing values retained",
                     donor_properties=train[["rho", "fc", "E28", "anchor_day"]].replace({np.nan: None}).to_dict("records"),
                     donor_groups=train.group.tolist(),
                     donor_times=[observations[c][0].tolist() for c in train.curve_id])
        diagnostics, cv = [], {}
        for family in FAMILIES:
            print(f"Fold {fold}/{folds}: fitting {family} on {len(train)} curves", flush=True)
            state, rows, result = fit_family(train, observations, family, scaler, rng, 64)
            prior["families"][family] = state
            diagnostics.extend(rows)
            cv[family] = result
        dump_json(fold_dir / "prior.json", prior)
        frame.to_csv(fold_dir / "split_manifest.csv", index=False)
        export_real(fold_dir / "real_observations.csv", frame, observations)
        pd.DataFrame(diagnostics).to_csv(fold_dir / "training_curve_fits.csv", index=False)
        details = {s: dict(curves=len(g), sources=g.group.nunique(), curve_ids=g.curve_id.tolist(),
                           source_ids=sorted(g.group.unique())) for s, g in frame.groupby("split")}
        assert all(not (set(details[a]["source_ids"]) & set(details[b]["source_ids"]))
                   for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
        summary["folds"].append(dict(fold=fold, partitions=details,
                                     prior_sha256=sha256(fold_dir / "prior.json"),
                                     conditional_link_diagnostics=cv))
        all_outer.extend(details["test"]["curve_ids"])
    assert sorted(all_outer) == sorted(meta.curve_id.tolist()), "Outer folds must cover every eligible curve once"
    dump_json(out / "fold_manifest.json", summary)
    print(json.dumps({"outer_folds": folds, "covered_curves": len(all_outer)}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    build(args.data_dir, args.out, args.seed, args.folds)


if __name__ == "__main__":
    main()

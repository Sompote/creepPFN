"""Predict additional compliance from an observed prefix and material properties."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import CreepPFN
from .pipeline import sha256, transform
from .train import predict, choose_device


def forecast(checkpoint_path, prior_path, times, observations, query_times,
             rho, fc, E28=np.nan, stress_ratio=np.nan, device="cpu"):
    times, observations, query_times = (np.asarray(v, float) for v in (times, observations, query_times))
    if times.ndim != 1 or observations.shape != times.shape or query_times.ndim != 1:
        raise ValueError("Times, observations and queries must be one-dimensional")
    if len(times) < 2 or len(query_times) < 1:
        raise ValueError("At least two context observations and one query are required")
    if not all(np.isfinite(a).all() for a in (times, observations, query_times)):
        raise ValueError("Context and query values must be finite")
    if np.any(np.diff(times) <= 0) or times[0] < 0 or times[0] > 10:
        raise ValueError("Context times must increase strictly and begin between day 0 and day 10")
    if np.any(query_times <= times[-1]) or np.any(query_times > 160):
        raise ValueError("Queries must follow all observations and be within day 160")
    if not (np.isfinite(rho) and rho > 0 and np.isfinite(fc) and fc > 0):
        raise ValueError("Density and strength must be finite and positive")
    if not (np.isnan(E28) or (np.isfinite(E28) and E28 > 0)):
        raise ValueError("E28 must be positive or missing (NaN)")
    if not (np.isnan(stress_ratio) or 0 < stress_ratio < 1):
        raise ValueError("Stress ratio must lie between 0 and 1 or be missing (NaN)")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if sha256(prior_path) != checkpoint["training_config"]["prior_sha256"]:
        raise ValueError("Prior hash differs from the training prior")
    prior = json.loads(Path(prior_path).read_text())
    meta = pd.DataFrame([dict(rho=rho, fc=fc, E28=E28, anchor_day=times[0], stress_ratio=stress_ratio)])
    record = dict(context_times=times, context_values=observations - observations[0],
                  query_times=query_times, targets=np.zeros_like(query_times),
                  features=transform(meta, prior["scaler"])[0])
    device = choose_device(device)
    model = CreepPFN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    median, low, high = predict(model, [record], device)[0]
    return pd.DataFrame(dict(t_day=query_times, increment_median=median,
                             increment_lower90=low, increment_upper90=high,
                             observation_scale_median=median + observations[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True, help="CSV with t_day and compliance columns")
    parser.add_argument("--query-days", type=float, nargs="+", required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--fc", type=float, required=True)
    parser.add_argument("--E28", type=float, default=float("nan"))
    parser.add_argument("--stress-ratio", type=float, default=float("nan"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.out.exists():
        raise FileExistsError(args.out)
    context = pd.read_csv(args.context)
    result = forecast(args.checkpoint, args.prior, context.t_day.to_numpy(), context.compliance.to_numpy(),
                      args.query_days, args.rho, args.fc, args.E28, args.stress_ratio)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

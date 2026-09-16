# Training and reproducibility

## What is included

The complete Python source used for generator fitting, synthetic task sampling,
training, fine-tuning, evaluation, ablation analysis, and SHAP analysis is in
`creep_prior/`. The processed five-fold data and fitted training-only priors are
in `data/folds/`. The 15 final fine-tuned checkpoints are in `models/`.

The processed data are sufficient to repeat synthetic pretraining and real-curve
fine-tuning from the saved fold priors. Re-fitting the priors from original NU
tables requires the three raw files expected by `creep_prior.make_folds` and is
documented in the source. Those upstream raw exports are not needed by the app.

## Reproduce one final member

From the repository root:

```bash
creep-pfn train \
  --fold fold_01 \
  --seed 45 \
  --tasks 5000 \
  --epochs 25 \
  --patience 6 \
  --device auto \
  --out runs/fold_01_seed45
```

This command reads the fold-specific model configuration and learning rates from
`data/architecture_choice.json`. Every pretraining epoch samples fresh synthetic
tasks from the fold's training-only prior. Fine-tuning then uses only real curves
marked `train`. Checkpoint selection uses only curves marked `validation`. The
outer-test partition is not loaded during fitting.

Repeat seeds 45, 46, and 47 for each of the five folds to reproduce the full set
of final members. Exact floating-point identity can depend on the accelerator,
PyTorch build, and deterministic-kernel support. Saved configurations, histories,
input hashes, and source partitions provide the audit trail.

## Protocol checks

```bash
creep-pfn verify
pytest
```

`verify` checks that every checkpoint references the packaged fold prior and that
training, validation, and test source identifiers do not overlap. The tests cover
future-target isolation, padding, chronology, generator reproducibility, grouped
ensemble intervals, and deployment inference.

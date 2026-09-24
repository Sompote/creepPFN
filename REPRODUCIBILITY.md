# Training and reproducibility

## What is included

The Python package `creep_prior/` contains the network, the training and evaluation code, and the subpackage `creep_prior/bayes/`, which implements the Kelvin4 curve family, the NumPyro fit of the hierarchical Bayesian prior and the sampler of synthetic creep tests. The processed five-fold data and the fitted priors are in `data/folds/`, the fold settings are in `data/training_settings.json`, and the 15 final fine-tuned networks are in `models/`. The scripts that reproduce the tables and figures of the paper, including the machine-learning baselines and the SHAP analysis, are in `research/` and are described in `research/README.md`.

## Fit a prior

The packaged priors can be refitted from the fold data with

```bash
pip install -e ".[research]"
python -m creep_prior.bayes.fit_prior --folds fold_01 --out runs/priors
```

which runs four NUTS chains of 1,500 warm-up and 500 retained draws on all training readings of the fold and stores the posterior draws with the discrepancy scale of 0.01 chosen on validation sources.

## Retrain one network

```bash
creep-pfn train --fold fold_01 --seed 45 --device auto --out runs/fold_01_seed45
```

This command reads the fold settings from `data/training_settings.json` and pretrains the network on 5,000 new synthetic tests per round drawn from the fold's prior, for at most 25 rounds with early stopping on the validation sources, before fine-tuning it on the measured training curves. The outer test sources are never loaded during fitting, and running seeds 45, 46 and 47 for all five folds (`sh research/train_all.sh`) reproduces the 15 networks, which together used 1,230,000 synthetic tests in the paper. Exact floating-point identity can depend on the accelerator and the PyTorch build, while the saved configurations, histories and input hashes provide the audit trail.

## Protocol checks

```bash
creep-pfn verify
pytest
```

`verify` checks that every checkpoint references its packaged fold prior and that the training, validation and test sources do not overlap, while the tests cover future-target isolation, padding, chronology, the synthetic-task sampler, ensemble intervals and deployment inference.

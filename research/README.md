# Research scripts

These scripts reproduce the main results of the paper (`paper/CreepPFN_paper.pdf`) from the files packaged in this repository. Outputs are written to `runs/`. Run them from the repository root after `pip install -e ".[research]"`.

| Paper item | Command | Notes |
|---|---|---|
| Hierarchical Bayesian prior (Sec. 3.3, Supplementary S3) | `python -m creep_prior.bayes.fit_prior --folds fold_01` | NUTS in NumPyro, 4 chains, 1,500 warm-up and 500 draws, Kelvin4 family, discrepancy scale 0.01. The fitted priors are already in `data/folds/*/prior.json`. |
| Training of the 15 networks (Sec. 3.4) | `sh research/train_all.sh` | Pretraining on synthetic tests from the prior, then fine-tuning on measured training curves. A GPU is recommended. |
| CreepPFN test scores (Table 4, Figure 3) | `python research/evaluate_test.py` | Reproduces `data/results/creeppfn_test_predictions.csv`. |
| Machine-learning baselines (Table 4) | `python research/ml_baselines.py` then `python research/score_models.py` | XGBoost, random forest and MLP, with and without the day-10 readings, tuned on validation sources. About one hour on a CPU. |
| SHAP analysis of the descriptors (Sec. 5.5, Figure 6) | `python research/shap_descriptors.py --device cuda` then `python research/analyze_shap.py` | Exact Shapley values over the descriptor groups, with each specimen's early readings held fixed. |

The external KMUTT and literature curves (Sec. 7.3) are not redistributed here (see `DATA_NOTICE.md`), so the external analysis cannot be rerun from this repository. Its results are recorded in `data/results.json`.

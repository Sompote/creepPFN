#!/bin/sh
# Retrain all 15 CreepPFN networks (5 folds x seeds 45-47) on the packaged hierarchical priors.
# Pretraining draws 5,000 synthetic tests per round from data/folds/<fold>/prior.json; about 1 min per network on a GPU.
set -e
for f in fold_01 fold_02 fold_03 fold_04 fold_05; do
  for s in 45 46 47; do
    creep-pfn train --fold "$f" --seed "$s" --device auto --out "runs/train/$f/seed$s"
  done
done

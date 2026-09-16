# Data notice

`data/folds/` contains processed records used by the reported source-disjoint
experiments. Each fold contains:

- `split_manifest.csv`: curve descriptors and train/validation/test membership;
- `real_observations.csv`: elapsed times and first-reading-referenced compliance;
- `prior.json`: the synthetic-task prior fitted only on that fold's training sources;
- `training_curve_fits.csv`: diagnostics for curves used to fit the generator.

The folds duplicate observations because each curve has a different role across
outer partitions. They are retained in this form so every checkpoint can be
verified against its exact prior and split. The original NU database should be
cited through Hubler, Wendner, and Bazant (2015), *ACI Materials Journal*, 112,
547–558, DOI: 10.14359/51687453. Users are responsible for confirming the terms
that apply to redistribution and reuse of the underlying database.

The KMUTT and separately labelled literature workbooks used for descriptive
external evaluation are not required to train the NU models and are not included
in this bundle. Their unresolved specimen-level publication mapping is documented
in the manuscript. No synthetic task is an independent experiment.

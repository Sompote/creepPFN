# CreepPFN supplementary application

CreepPFN forecasts concrete creep compliance from material descriptors and a
short, irregular measurement history. This repository provides two application
modes:

1. a command-line interface for reproducible batch prediction; and
2. a Gradio interface for interactive prediction and visualization.

It also contains the processed training data, training-only synthetic priors,
15 frozen checkpoints, training code, validation tests, and the manuscript PDF.

![CreepPFN workflow](assets/paper/model_workflow.png)

The illustrated description, quantitative evidence, interpretation, and limits
are provided in [MODEL_CARD.md](MODEL_CARD.md). The Gradio app also includes
Model card and Paper figures tabs.

## Installation

Python 3.10 or later is required. An editable installation keeps the large model
and data files outside the Python environment.

```bash
cd /path/to/CreepPFN-supplement
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
creep-pfn verify
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Command-line prediction

The context CSV must contain `t_day` and `compliance`. Compliance is measured in
microstrain/MPa. At least two readings are required. All context readings must be
available by day 10; a reading exactly on day 10 is not required.

```csv
t_day,compliance
1,10
3,15
7,20
10,23
```

Run the packaged 15-member exploratory deployment ensemble:

```bash
creep-pfn predict \
  --context examples/context.csv \
  --rho 2400 \
  --fc 40 \
  --E28 30000 \
  --query-days 14 28 56 90 120 160 \
  --fold all \
  --out predictions.csv \
  --plot forecast.png
```

Units are kg/m³ for density, MPa for compressive strength and elastic modulus,
elapsed days since loading for time, and microstrain/MPa for compliance. Omit
`--E28` when the modulus is unavailable. Use `--fold fold_01` through `fold_05`
to obtain one paper fold's three-member ensemble.

The model subtracts the first compliance reading internally. Increment columns
are relative to that anchor. Compliance columns add the anchor back. The reported
90% intervals are marginal at each requested time.

## Gradio interface

```bash
creep-pfn app --inbrowser
```

Open `http://127.0.0.1:7860` if the browser does not open automatically. The
equivalent direct command is `python app.py`. Model loading occurs on the first
prediction and is cached for later requests.

## Training

The saved source-disjoint partitions and priors allow a final model member to be
retrained without importing the original database again:

```bash
creep-pfn train --fold fold_01 --seed 45 --out runs/fold_01_seed45
```

The full settings and protocol are described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
Training the complete 15-member ensemble is computationally expensive and is
intended for a CUDA GPU. The CLI and Gradio app can run on CPU, Apple MPS, or CUDA.

## Repository contents

```text
creep_prior/       model, generator, training, evaluation, SHAP, and app code
data/folds/        processed observations, source splits, and fitted priors
models/            15 final fine-tuned checkpoints
assets/paper/      workflow, architecture, data, result, SHAP, and forecast figures
examples/          example context input
tests/             scientific and software checks
paper/             current manuscript PDF
```

Read [MODEL_CARD.md](MODEL_CARD.md) before interpreting predictions and
[DATA_NOTICE.md](DATA_NOTICE.md) before redistributing the processed data. This is
a research model. It does not replace creep testing, constitutive assessment, or
structural engineering review.

The hashes of the frozen checkpoints, fitted priors, reported results, and paper
are recorded in `MANIFEST.sha256`. On macOS or Linux, verify them with
`shasum -a 256 -c MANIFEST.sha256`.

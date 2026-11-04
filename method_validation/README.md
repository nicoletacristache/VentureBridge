# Method Validation

Frozen benchmark inputs, measurement-comparison implementations, ECD encoders, saved predictions, and result tables for VentureBridge.

Read [PROVENANCE.md](PROVENANCE.md) for the dataset source, reference-value interpretation, and evaluation scope. This note is the authoritative source description for this folder.

## Contents

- `datasets/benchmark_data.zip`: 36 frozen input and protocol assets with their integrity metadata.
- `analysis/`: three method/evaluation scripts and pinned analysis dependencies.
- `results/`: 22 saved metric tables, summaries, and figures.
- `reference_outputs/frozen_predictions.zip`: nine final prediction and configuration files.
- `source_manifest.json`: hashes of original assets.
- `prepare_assets.py`: unpacks and verifies the archives without silently overwriting different local files.

The archives keep individual upload sizes below GitHub's browser limit and preserve the original file bytes. They do not contain developer conversations or credentials.

## Setup and archived-result inspection

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r method_validation/analysis/requirements_measurement_validation.txt
python method_validation/prepare_assets.py --with-predictions
```

Saved summary tables and figures are directly available under `results/`. The preparation command restores `data/` and, optionally, `predictions/` under this folder and checks every extracted file against `source_manifest.json`.

## Measurement methods

```bash
python method_validation/analysis/run_measurement_validation.py
```

The runner contains equal-weight ECD aggregation, final-resource-state scoring, raw rubric mean, CTT anchor equating, static multifacet, dynamic without anchors, simplified anchored dynamic, the full method, and an Oracle reference. Definitions and applicability are documented in [PROVENANCE.md](PROVENANCE.md).

Rerunning this command writes new local outputs to `method_validation/predictions/` and `method_validation/results/`. Preserve the published checkout before replacing its saved results.

## ECD coding comparison

To evaluate the archived predictions without a model call:

```bash
python method_validation/analysis/evaluate_ecd_audit.py
```

To create new rule predictions:

```bash
python method_validation/analysis/score_ecd_audit.py --method rule
```

New LLM predictions require privately configured `V_API_KEY` and an HTTPS `V_API_BASE_URL`. The preserved inference setup uses `qwen3.5-flash`. The inference script imports `fcntl`, so Windows users need a POSIX-compatible environment. Model calls send selected benchmark input to the configured provider. Do not commit local credentials.

The preparation script and archived-prediction evaluation do not call a model service. New inference is separate from inspecting the published results.

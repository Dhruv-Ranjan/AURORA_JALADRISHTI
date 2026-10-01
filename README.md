# AURORA-SR reproducibility release

This folder contains the publishable, separable materials for the synthetic
AURORA mission-scheduling experiment described in the AURORA-SR manuscript.
The manuscript itself is intentionally not included.

## Contents

- `aurora_publication_ready.py` — simulator and matched-seed experiment runner.
- `Physics.py` — orbital, access, eclipse, and power-model utilities.
- `results/` — configuration, machine-readable results, decision records,
  figures, and the executed experiment report.
- `validation_artifacts/` — selected summary artifacts for the related
  validation runs.

## Running the experiment

Install Python 3.10+ with NumPy, Matplotlib, and SciPy, then run:

```text
python aurora_publication_ready.py --quick
python aurora_publication_ready.py
```

The quick run is only a smoke test. The full run writes a new results
directory and may take substantially longer.

## Deliberate exclusions

The release does not include the manuscript, the 1.1 GB cached drought
dataset, model checkpoints, private databases, caches, or the broader
training and application source tree. The validation JSON files are included
only as compact result summaries and are not a substitute for the withheld
training artifacts.

The simulator is a synthetic evaluation. Its results should be interpreted
only under the assumptions and scenarios recorded in
`results/experiment_config.json` and `results/research_report.md`.

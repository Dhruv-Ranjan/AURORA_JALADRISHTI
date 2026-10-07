from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "README.md",
    "PROVENANCE.md",
    "requirements.txt",
    "requirements-research.txt",
    "Physics.py",
    "aurora_publication_ready.py",
    "remote_training/dataset.py",
    "remote_training/train_remote.py",
    "remote_training/regions.example.json",
    "results/experiment_config.json",
    "results/research_report.md",
    "results/summary.csv",
    "validation_artifacts/inference_validation_results.json",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing release files: " + ", ".join(missing))

    config = json.loads((ROOT / "results/experiment_config.json").read_text(encoding="utf-8"))
    runs = config["config"]["monte_carlo_runs"]
    scenarios = config["scenarios"]
    if runs != 24 or len(scenarios) != 4:
        raise SystemExit("Unexpected experiment configuration")

    forbidden_suffixes = {".db", ".npz", ".pt", ".pth", ".pyc"}
    forbidden = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if forbidden:
        raise SystemExit("Generated or private files present: " + ", ".join(map(str, forbidden)))

    print("AURORA-SR release verification passed")


if __name__ == "__main__":
    main()

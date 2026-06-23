# Phase 0: Setup

**Date:** 2026-06-24 05:40 KST
**Duration:** ~0.5 hours
**Phase outcome:** COMPLETE
**Hour of total budget consumed:** ~0.5 / 36

---

## 1. What was done

The repository skeleton from PROJECT_PLAN.md was created in full: src, notebooks, figures, report/sections, experiments/checkpoints, data/raw, data/processed, and docs/journal. A Python 3.13.4 virtual environment was created at .venv (per the user's request that nothing install globally) and all dependencies from requirements.txt installed cleanly into it, including torch 2.12.1, arch 8.0.0, statsmodels 0.14.6, mlflow 3.14.0, pandas 2.3.3, and numpy 2.5.0. Local git was initialized with one commit; the GitHub push was intentionally deferred to Phase 6 per the user's instruction. The Phase 0 success check (`import torch, arch, yfinance, mlflow`) returns ok. MLflow was verified end to end by creating an experiment, logging a parameter, and tearing it down. The data ingestion module src/data.py was also written in this phase (it is a Phase 1 deliverable but was authored here while dependencies installed) and is ready to run.

---

## 2. Decisions made

**Decision:** Use a SQLite MLflow tracking backend (`sqlite:///experiments/mlflow.db`) instead of the plain file store (`./experiments/mlruns`) named in the plan.
**Alternatives:** Keep the file store via the `MLFLOW_ALLOW_FILE_STORE=true` escape hatch.
**Reason:** MLflow 3.14 places the file store in maintenance mode and refuses it by default; SQLite is the recommended, non-deprecated backend and is strictly better for the reproducibility principle.

**Decision:** Contain all installs in a repo-local venv at .venv.
**Alternatives:** Install into the global homebrew Python 3.13.
**Reason:** The user explicitly asked that nothing land outside the repo.

**Decision:** Defer the GitHub remote and push.
**Alternatives:** Create a private repo now as the plan's Phase 0 step 2 suggests.
**Reason:** User instruction; the plan keeps the repo private until Phase 6 anyway, so nothing is lost.

---

## 3. Results

### Quantitative

| Component | Version / status |
|-----------|------------------|
| Python    | 3.13.4 (arm64)   |
| torch     | 2.12.1           |
| arch      | 8.0.0            |
| statsmodels | 0.14.6         |
| mlflow    | 3.14.0           |
| pandas    | 2.3.3            |
| numpy     | 2.5.0            |
| Import success check | ok    |
| MLflow backend | sqlite:///experiments/mlflow.db, verified |

### Qualitative

**Surprise:** MLflow 3.14 hard-deprecated the filesystem tracking backend that the plan assumed. This is a clean one-line fix (switch the tracking URI to SQLite) but every later phase that logs runs must use the SQLite URI, not `./experiments/mlruns`.

---

## 4. Report prose draft

> All experiments were tracked with MLflow \cite{mlflow} using a local SQLite backend, with seeds fixed and logged per run to support exact reproduction. The software stack was Python 3.13 with PyTorch \cite{pytorch} for the neural models, the arch package for GARCH estimation, and statsmodels for the HAR-RV regression. The full environment is pinned in requirements.txt and the data pipeline is regenerable end to end from a single entry point.

### Citations needed for this prose

- mlflow: Zaharia et al., "Accelerating the Machine Learning Lifecycle with MLflow", IEEE Data Eng. Bull., 2018.
- pytorch: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS, 2019.

---

## 5. Figures generated

None this phase.

---

## 6. Files created or modified

```
requirements.txt (new, 14 lines)
.gitignore (new, 37 lines)
src/data.py (new, 240 lines)        # Phase 1 deliverable, authored early
docs/journal/phase_0_setup.md (new)
.venv/ (new, gitignored)
experiments/mlflow.db (new, gitignored)
directory tree: src, notebooks, figures, report/sections,
  experiments/checkpoints, data/raw, data/processed, docs/journal
```

---

## 7. AI tool usage in this phase

**Tool:** Claude Code (claude-opus-4-8)
**Task:** Created the repo skeleton, requirements.txt, .gitignore, the venv, and the dependency install; authored src/data.py.
**Verification:** Ran the plan's import success check (returns ok); printed and recorded every key package version; smoke-tested the MLflow SQLite backend by creating, logging to, and deleting an experiment; inspected git status to confirm .venv, data/, and experiments artifacts are correctly gitignored.

---

## 8. Risk register

**Next phase risk:** The Oxford-Man Realized Library was discontinued from public hosting in 2022, so the primary data path will likely fail and trigger DECISION POINT 1 (fall back to yfinance + Garman-Klass).
**Final report risk:** The MLflow backend change must be reflected consistently in any "reproduce" instructions so a grader can rerun the pipeline.
**Submission risk:** None at this stage.

---

## 9. Lessons

The plan was written against MLflow's older file-store default; the 3.14 release removed it. Pinning exact versions in requirements.txt would have surfaced this earlier, but the one-line SQLite switch is harmless and arguably an upgrade. No time lost.

---

## 10. Prep for next phase

1. Attempt `python src/data.py oxford` to test the Oxford-Man path; expect failure and surface DECISION POINT 1.
2. On the expected fallback, run `python src/data.py yfinance` to pull SPX, FTSE, N225, DAX and persist parquet.
3. Build notebooks/01_eda.ipynb and save at least two figures to figures/.

# AI Tool Usage Log

This log is required by the IE412 rubric. Every interaction with an AI tool that contributed to the submitted work is logged here. The report's appendix is built from this file.

## Format

For each entry:

```
### YYYY-MM-DD HH:MM, Phase N
**Tool:** <name and version>
**Task:** <what was generated or asked>
**Verification:** <how the output was checked>
```

---

## Entries

### 2026-06-23 (kickoff), Phase -1 (planning)
**Tool:** Claude (claude-opus-4-7) via web interface
**Task:** Brainstormed project ideas from the IE412 brief, scoped the multi-timescale volatility project, produced `PROJECT_PLAN.md` and the `phase-documentation` skill.
**Verification:** Reviewed scope against the rubric (problem definition, motivation, method, data, results, discussion). Cross-checked the 36-hour budget against the actual deadline. Plan to verify each implementation phase as it's built.

### 2026-06-24 05:40, Phase 0 (setup)
**Tool:** Claude Code (claude-opus-4-8)
**Task:** Built the repo skeleton, requirements.txt, .gitignore, a repo-local venv, installed all dependencies, initialized local git, and authored src/data.py.
**Verification:** Ran the import success check (`import torch, arch, yfinance, mlflow` -> ok), recorded all key package versions, smoke-tested the MLflow SQLite backend (create/log/delete an experiment), and confirmed via git status that environment and data artifacts are gitignored.

### 2026-06-24 05:45, Phase 1 (data)
**Tool:** Claude Code (claude-opus-4-8)
**Task:** Implemented src/data.py (Oxford-Man and yfinance Garman-Klass loaders, cleaning, train/val/test splits, summary stats) and notebooks/01_eda.ipynb; built parquet for SPX/FTSE/N225/DAX and two EDA figures.
**Verification:** Probed both data sources before selecting one; confirmed all four assets' lag-1 autocorrelations fall in the plan's [0.5, 0.8] sanity band; visually inspected the rendered time-series and ACF/PACF figures; checked split-size arithmetic against the date boundaries.


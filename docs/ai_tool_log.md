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

### 2026-06-24 06:10, scope change (MEDIUM CUT)
**Tool:** Claude Code (claude-opus-4-8)
**Task:** Edited PROJECT_PLAN.md to reduce scope to the MEDIUM CUT after DECISION POINT 2. The working budget was cut from ~36 to ~10 hours; the project narrowed to SPX only; the Vanilla LSTM and the vol-targeting backtest were dropped; the model set is now GARCH(1,1), HAR-RV, and the dual-resolution TCNN-LSTM; evaluation is statistical only (RMSE, QLIKE, Diebold-Mariano); the report target dropped to 5 to 6 pages.
**Reason:** At DECISION POINT 2 the HAR-RV validation log-RMSE (0.77 to 0.90) sat above the plan's [0.3, 0.6] band, but diagnostics showed the models were correct (textbook coefficients, HAR beats climatology and random walk on every split) and the gap was caused by the noisier Garman-Klass daily proxy replacing the discontinued Oxford-Man 5-minute RV. Combined with a tightened time budget, the user elected the MEDIUM CUT to ship a focused, clean single-asset statistical comparison rather than a broader study at risk against the deadline.
**Verification:** Applied each specified edit by exact-string match and grepped the file afterward for residual out-of-scope terms (backtest, Sharpe, four models, multi-seed, hyperparameter sweep); two Vanilla LSTM references outside sections 3a/3c were left in place pending user confirmation and flagged.

### 2026-06-24 06:20, Phase 2 (baselines)
**Tool:** Claude Code (claude-opus-4-8)
**Task:** Wrote src/evaluate.py (RMSE, Patton QLIKE) and src/baselines.py (HAR-RV OLS, GARCH(1,1) with train-only level calibration), fitted both for SPX, and logged coefficients and metrics to the MLflow "baselines" experiment. After the MEDIUM CUT, scoped the run to SPX and regenerated a clean store.
**Verification:** Confirmed HAR-RV slopes are positive and sum to 0.93 (Corsi regularity) and GARCH persistence is 0.98; ran a benchmark diagnostic (HAR beats climatology and random walk on every split) to establish the RMSE floor is proxy noise; checked QLIKE non-negativity; confirmed MLflow holds exactly the two expected SPX runs.

### 2026-06-24 06:45, Phase 3 (neural)
**Tool:** Claude Code (claude-opus-4-8)
**Task:** Implemented src/models.py (DualResTCNNLSTM ported from the ADDIM HAR architecture) and src/train.py (windowing, train-only standardisation, Adam + cosine schedule, early stopping, MLflow logging, checkpoint), trained the model on SPX with seed 42, and produced the training-curves figure.
**Verification:** Checked the parameter count and that padding preserves window length; confirmed train loss decreased and early stopping selected the best validation epoch (27); scored val/test with the same evaluate.py used for baselines so the information set and target match exactly; inspected the training-curve figure.

### 2026-06-24 07:20, Phase 4 (evaluation)
**Tool:** Claude Code (claude-opus-4-8)
**Task:** Implemented the Diebold-Mariano test in evaluate.py, ran pairwise DM tests across the three models, generated the four report figures (headline analogy, predictions, metrics table, robustness/ablation), and added two evaluation-driven experiments beyond the cut: a 5-seed robustness sweep and a short/long branch ablation.
**Verification:** Checked DM statistic signs against the metric ordering and QLIKE non-negativity; confirmed all five robustness seeds reuse the identical pipeline and land within a 0.005 RMSE band (5/5 beat HAR-RV on RMSE, 1/5 on QLIKE), which corrected an overstated single-seed QLIKE claim; verified the ablation variants differ only in active branches; visually inspected every figure.


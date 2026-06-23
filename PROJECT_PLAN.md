# IE412 Term Project: Multi-Timescale Volatility Forecasting

**Author:** Mohamed Diallo
**Course:** IE412 AI for Finance, UNIST Spring 2026
**Deadline:** June 24, 2026, 24:00 KST
**Working budget:** ~36 hours from kickoff

---

## North star (one paragraph)

Corsi's HAR-RV model (2009) forecasts realized volatility by linearly combining daily, weekly, and monthly RV averages. That hand-engineered multi-timescale decomposition is conceptually identical to the dual-resolution TCNN-LSTM I published for Human Activity Recognition (CEA 2026), which combines a short-context branch (2 minutes) and a long-context branch (2 hours). This project transfers that architecture to financial volatility forecasting. Four models compete: GARCH(1,1), HAR-RV, vanilla LSTM, and the dual-resolution TCNN-LSTM. Evaluation is both statistical (RMSE, QLIKE, Diebold-Mariano tests) and decision-focused (vol-targeting backtest with Sharpe and drawdown). The narrative ties prediction quality to decision quality, which strengthens the AOS application and reuses my published architecture in a new domain.

---

## Operating principles for the agent (read every session)

1. This file is the spec. Re-read the relevant phase before acting.
2. **Stop after every phase.** Do not proceed until: (a) the `phase-documentation` skill has been invoked and a journal entry exists at `docs/journal/phase_N_<name>.md`, and (b) the user has reviewed and approved.
3. **Stop at every decision point** (marked `DECISION POINT`). Ask the user; never pick autonomously.
4. **Cut list discipline.** If time pressure hits, consult the cut list at the bottom. Do not add features that are not in this plan.
5. **Style.** Concise output, no em dashes, no bullet-heavy formatting, copy-paste-ready code, natural prose in explanations. Match this style in journal entries and report drafts.
6. **Reproducibility.** Every experiment goes through MLflow. Seeds fixed and logged. Data version recorded.
7. **AI tool log.** Keep `docs/ai_tool_log.md` updated as we go (the report requires this).

---

## Success criteria

A submission counts as successful if all of the following hold:

- A 6 to 8 page PDF report is submitted by June 24, 22:00 KST (2 hour safety buffer).
- The report contains all six rubric sections: problem definition, motivation, proposed method, data and implementation, results, discussion.
- At least three of the four models produce reportable numbers on the test set.
- At least one statistical evaluation (RMSE or QLIKE) and one decision-focused evaluation (backtest) are reported.
- The conceptual analogy between HAR-RV and the TCNN-LSTM is made explicit with a figure.
- The GitHub repo is public with a README that leads with figures.

A successful submission does **not** require the deep model to beat HAR-RV. Negative results are explicitly allowed by the rubric and have a discussion script ready (see Phase 4).

---

## Repo structure (create in Phase 0)

```
volatility-project/
├── PROJECT_PLAN.md              # this file
├── README.md                    # public-facing, fill in Phase 6
├── requirements.txt
├── .gitignore
├── .claude/
│   └── skills/
│       └── phase-documentation/
│           └── SKILL.md
├── data/
│   ├── raw/                     # downloaded files, gitignored
│   └── processed/               # parquet outputs, gitignored
├── src/
│   ├── data.py                  # ingestion, RV computation
│   ├── baselines.py             # GARCH, HAR-RV
│   ├── models.py                # LSTM, TCNN-LSTM
│   ├── train.py                 # training loop with MLflow
│   ├── evaluate.py              # RMSE, QLIKE, DM tests
│   └── backtest.py              # vol-targeting backtest
├── notebooks/
│   └── 01_eda.ipynb
├── experiments/                 # MLflow artifacts
├── figures/                     # SVG/PDF outputs for report
├── report/
│   ├── main.tex
│   ├── refs.bib
│   └── sections/
└── docs/
    ├── journal/                 # one file per phase
    │   └── _TEMPLATE.md
    └── ai_tool_log.md
```

---

# PHASES

Each phase has: **Objective**, **Tasks**, **Deliverables**, **Success check**, **Decision points**, **Stop for documentation**.

---

## Phase 0: Setup (1 hour)

**Objective.** Repo skeleton, environment, MLflow running, skill installed.

**Tasks.**
1. Create the directory tree above.
2. Initialize git, push to a new private GitHub repo (will flip to public in Phase 6).
3. Create `requirements.txt` with: `numpy pandas scipy scikit-learn torch arch statsmodels yfinance pyarrow mlflow matplotlib seaborn pytest jupyter`. Pin `torch` to the CPU build unless on Colab.
4. Install the `phase-documentation` skill (already at `.claude/skills/phase-documentation/`).
5. Initialize MLflow with `mlflow ui` test; tracking URI `./experiments/mlruns`.
6. Create `docs/ai_tool_log.md` and put the first entry (Claude Code, project planning).

**Deliverables.** Working repo, clean `pip install -r requirements.txt`, MLflow UI loads.

**Success check.** `python -c "import torch, arch, yfinance, mlflow; print('ok')"` returns ok.

**Stop for documentation.** Invoke `phase-documentation` skill, write `docs/journal/phase_0_setup.md`.

---

## Phase 1: Data (2 to 3 hours, HARD CAP 4 HOURS)

**Objective.** Clean daily log realized variance series for 3 to 4 major indices, split into train/val/test, persisted to parquet.

**Primary plan.** Oxford-Man Institute Realized Library (5-min pre-computed RV, free academic access). Pull SPX (.SPX), FTSE (.FTSE), N225 (.N225), DAX (.GDAXI). Use the 5-min RV series (`rv5` column).

**Fallback plan.** If Oxford-Man is down, schema-broken, or auth-walled by hour 1.5, switch to `yfinance` daily OHLC and compute Garman-Klass range-based volatility as RV proxy. Document the substitution clearly.

**Tasks.**
1. Implement `src/data.py` with two functions: `load_oxford_man(ticker)` and `load_yfinance_gk(ticker)`. Both return a DataFrame with columns `[date, rv, log_rv]`.
2. EDA notebook `notebooks/01_eda.ipynb`: plot log-RV time series per asset, ACF/PACF, basic stats (mean, std, skew, kurt, persistence). Save 2 plots to `figures/`.
3. Define splits: train 2000-01-01 to 2018-12-31, val 2019-01-01 to 2020-12-31 (COVID stress test in val by design), test 2021-01-01 to 2025-12-31.
4. Persist to `data/processed/<ticker>.parquet`.

**Deliverables.** Parquet files, EDA notebook with at least 2 saved figures, summary stats table for the report.

**Success check.** Log-RV series have no NaNs after cleaning, no extreme outliers (z > 8 should be inspected, not blindly dropped). Persistence (lag-1 AC) should be in roughly [0.5, 0.8] for daily log-RV; if it's near zero, the RV computation is wrong.

**DECISION POINT 1 (hour 1.5).** If Oxford-Man isn't loading cleanly, switch to yfinance + Garman-Klass. Ask user.

**Stop for documentation.** Write `docs/journal/phase_1_data.md`. Include the EDA figures and the summary stats table. Draft 3 to 5 sentences of report prose for the "Data" section.

---

## Phase 2: Classical baselines (2 to 3 hours)

**Objective.** GARCH(1,1) and HAR-RV fitted per asset, val and test numbers logged in MLflow.

**Tasks.**
1. `src/baselines.py`:
   - `fit_garch(train_returns)` using `arch` package, returns fitted model. Forecast next-day variance, transform to log-RV space for comparison.
   - `fit_har_rv(train_log_rv)` does OLS of `log_rv[t+1]` on `log_rv[t]`, `mean(log_rv[t-4:t+1])`, `mean(log_rv[t-21:t+1])`. Returns statsmodels result.
2. `src/evaluate.py`: `rmse(y, yhat)` and `qlike(y_rv, yhat_rv)` (QLIKE on RV space, not log; standard convention).
3. Run both baselines per asset, log all coefficients and metrics to MLflow under experiment name `baselines`.

**Deliverables.** MLflow experiment with baseline runs. Coefficient tables for HAR-RV per asset (these go in the appendix or as a table in the report).

**Success check.** HAR-RV log-RMSE on validation should be roughly in [0.3, 0.6] for major indices. If it's much larger, RV computation is wrong, fix in Phase 1 before continuing. HAR-RV coefficients should be positive and roughly sum to 1 (Corsi's empirical regularity).

**DECISION POINT 2.** If HAR-RV breaks the sanity bounds, stop and debug Phase 1. Do not move to neural models with broken data.

**Stop for documentation.** Write `docs/journal/phase_2_baselines.md`. Include coefficient tables. Draft 4 to 6 sentences of "Baselines" report prose explaining HAR-RV and the daily/weekly/monthly intuition (this prose is the bridge to the analogy figure).

---

## Phase 3: Neural models (6 to 8 hours)

**Objective.** Vanilla LSTM and dual-resolution TCNN-LSTM trained, val numbers in MLflow, best checkpoints saved.

**Tasks.**

### 3a: Architecture port
1. `src/models.py`:
   - `VanillaLSTM(input_size=1, hidden=64, layers=2, dropout=0.2)` taking 22-day windows.
   - `DualResTCNNLSTM(short_window=5, long_window=22, tcnn_channels=[32,64], lstm_hidden=64)`. Two TCNN branches (1D dilated conv blocks, kernel 3, dilations [1,2,4]), each followed by global avg pool, concatenated, fed to a single LSTM cell with one timestep (treat the concatenation as a sequence of length 1), then MLP head outputting scalar log-RV.

Port the `HybridTCNNLSTM` class from the ADDIM repo as the starting point. Adapt: input channels go from sensor count to 1, output goes from class logits to regression scalar, loss changes from CE to MSE.

### 3b: Training loop
2. `src/train.py`: standard PyTorch loop, Adam optimizer (lr 1e-3 with cosine schedule), early stopping on val RMSE (patience 15), batch size 64, max 200 epochs. MLflow logs: hyperparams, train/val loss per epoch, final metrics, model artifact.
3. Pool all assets into a single training set (one model trained on SPX+FTSE+N225+DAX). Asset-specific models are a stretch goal in the cut list.

### 3c: Runs
4. Vanilla LSTM: 3 seeds (42, 1337, 2024).
5. Dual-res TCNN-LSTM: 3 seeds, same seeds for fair comparison.
6. Quick hyperparam check (not a full sweep): try 2 values of `lstm_hidden` (32, 64) and 2 dropout values (0.1, 0.3) for the TCNN-LSTM only. 4 configs × 3 seeds = 12 runs, roughly 90 minutes on CPU, faster on a GPU if available.

**Deliverables.** MLflow experiment `neural`, best checkpoints in `experiments/checkpoints/`.

**Success check.** Vanilla LSTM val log-RMSE should be roughly competitive with HAR-RV (within ±0.05). TCNN-LSTM should be at least competitive; "beats HAR-RV" is the hope but not the success bar.

**DECISION POINT 3 (hour 14 of total budget).** If by hour 14 no neural model is training cleanly, cut seed averaging (run single seed) and the hyperparam check. If by hour 16 still nothing, drop the TCNN-LSTM and submit with three models (GARCH, HAR-RV, LSTM). Ask user.

**Stop for documentation.** Write `docs/journal/phase_3_neural.md`. Include training curves figure. Draft 6 to 8 sentences of "Method" report prose explicitly drawing the HAR-RV / TCNN-LSTM analogy. This is the most important prose draft of the project.

---

## Phase 4: Evaluation (3 to 4 hours)

**Objective.** Statistical comparison with DM tests, vol-targeting backtest, all figures generated for the report.

**Tasks.**
1. `src/evaluate.py` extended with `diebold_mariano(e1, e2, h=1, loss='se')`. Run pairwise DM tests across all four models per asset, build a significance matrix.
2. `src/backtest.py`: vol-targeting strategy. Daily, position size = `target_vol / predicted_vol`, capped at 2x leverage, with a 5 bps cost per turnover. Compute equity curve, Sharpe, max drawdown, turnover, hit rate. Same horizon as test set.
3. Figures (save as both SVG and PDF for LaTeX):
   - `fig_har_analogy.svg`: side-by-side diagram of HAR-RV decomposition (daily/weekly/monthly lags) and the dual-resolution TCNN-LSTM branches. **This is the headline figure.** Draw it in matplotlib or just in `tikz` inside the LaTeX if faster.
   - `fig_predictions.svg`: predicted vs actual log-RV for one asset, test period, all four models on one plot.
   - `fig_equity_curves.svg`: backtest equity curves for all four strategies + buy-and-hold baseline.
   - `fig_metrics_table.svg` or LaTeX table: RMSE, QLIKE, Sharpe, MDD per model per asset.
4. Optional ablation if time allows: short-only branch vs long-only branch vs dual (mirror the ADDIM paper exactly). Cut first if time pressed.

**Deliverables.** All figures in `figures/`, results dictionary in `experiments/final_results.json`.

**Success check.** Numbers stay reasonable (no Sharpe = 50, no negative QLIKE). Sanity check: buy-and-hold Sharpe on SPX 2021-2025 should be roughly 0.5 to 1.0.

**DECISION POINT 4.** If the backtest produces weird results (e.g. extreme leverage swings, negative Sharpe across all models), drop the backtest entirely and report stats-only. Better to ship a clean econometric study than a broken backtest. Ask user.

**Negative result script.** If the TCNN-LSTM does not beat HAR-RV on RMSE, the discussion frames this as: "Corsi's linear decomposition is already near-optimal at daily horizons because the underlying signal is approximately log-Gaussian with persistence cleanly captured by three lag aggregates. The multi-timescale prior matters more when the relationship is nonlinear, which suggests intraday horizons, regime-switching periods, or multi-asset cross-section as more promising application domains. The architecture's value in the HAR domain came from nonlinear sensor interactions; financial volatility may not have enough nonlinearity at this horizon to reward the inductive bias." Use this verbatim if needed.

**Stop for documentation.** Write `docs/journal/phase_4_evaluation.md`. Embed all figures. Draft the full "Results" section prose (the longest single draft, aim for one full page of LaTeX).

---

## Phase 5: Report (6 to 8 hours)

**Objective.** Submit a 6 to 8 page PDF by June 24, 22:00 KST.

**Tasks.**
1. Copy LaTeX template from the CEA 2026 paper.
2. Assemble sections from the journal prose drafts. The journal entries should make this assembly, not writing.
3. Sections and target lengths:
   - Abstract (150 words)
   - Introduction and motivation (1 page)
   - Related work and the HAR-RV / HAR analogy (0.75 page, includes `fig_har_analogy`)
   - Method (1.25 pages, includes model architecture details)
   - Data and implementation (0.75 page)
   - Results (2 pages, includes 3 figures and the metrics table)
   - Discussion and limitations (0.75 page)
   - Conclusion (0.25 page)
   - References
   - Appendix: hyperparameters, additional figures, AI tool usage statement
4. AI tool usage statement at the end, in the appendix. Pull from `docs/ai_tool_log.md`. Be specific: "Claude Code (Sonnet 4.6) was used for the training loop scaffold; outputs were verified by reviewing the loss curves and reproducing one run by hand."
5. Polish pass: spellcheck (en-US), check all citations resolve, all figures have captions, all tables have captions, no orphan widows.

**Deliverables.** `report/main.pdf` ready for submission.

**Success check.** PDF compiles. Page count between 6 and 8 (excl. refs and appendix). All rubric sections present. Compare against rubric checklist one more time.

**Stop for documentation.** Write `docs/journal/phase_5_report.md` capturing what made it into the final draft and what was cut.

---

## Phase 6: Repo polish and submission (1 to 2 hours)

**Objective.** Public GitHub repo, README with figures, submission complete.

**Tasks.**
1. Flip repo to public.
2. README.md leads with the headline figure (`fig_har_analogy`), then a 3-sentence summary, then results table, then reproduction instructions.
3. Add a "Reproduce" section: `pip install -r requirements.txt`, `python src/data.py`, `python src/train.py`, `python src/evaluate.py`.
4. Tag a release `v1.0-submission`.
5. Link the repo URL inside the report's front matter or appendix.
6. Submit the PDF on the course platform.

**Deliverables.** Public repo, tagged release, PDF submitted.

**Stop for documentation.** Write `docs/journal/phase_6_submission.md`. Brief retrospective: what went well, what would I do differently, what is the next experiment.

---

# DECISION TREE SUMMARY

```
Hour 1.5: Data source confirmed (Oxford-Man or yfinance+GK)
Hour 8:   HAR-RV sanity check passes
Hour 14:  Neural models training (else cut seed avg + hyperparam)
Hour 16:  At least one neural model done (else drop TCNN-LSTM)
Hour 22:  Evaluation complete, figures done
Hour 32:  Report draft complete
Hour 36:  Submitted (target 22:00 KST = hour 34 if started morning of June 23)
```

---

# CUT LIST (in order)

When time slips, cut in this order. Do not improvise; follow this list.

1. Streamlit dashboard
2. Ablation study (short-only / long-only / dual)
3. Hyperparameter mini-sweep (run with defaults instead)
4. Seed averaging (run with seed 42 only)
5. Vol-targeting backtest (report stats-only)
6. Two of the four assets (keep SPX + one other)
7. TCNN-LSTM model (submit with GARCH + HAR-RV + LSTM and frame as "deep model deferred")

---

# AI TOOL LOG (mandatory per rubric)

Maintained at `docs/ai_tool_log.md`. Update after every phase. Format per entry:

- Tool name and version
- Phase
- Specifically what it was used for
- How outputs were verified

The report's appendix pulls from this verbatim.

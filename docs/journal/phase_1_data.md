# Phase 1: Data

**Date:** 2026-06-24 05:45 KST
**Duration:** ~0.5 hours
**Phase outcome:** COMPLETE
**Hour of total budget consumed:** ~1.0 / 36

---

## 1. What was done

Daily log realized-variance series were built for SPX, FTSE, N225, and DAX over 2000 to 2026 and persisted to data/processed/<TICKER>.parquet with train/val/test split labels. The primary Oxford-Man Realized Library path returned HTTP 404 (Oxford discontinued public hosting in 2022 and the known mirror is gone), so per DECISION POINT 1 and the user's confirmation the project switched to the planned fallback: daily OHLC from yfinance with a Garman-Klass range-based variance estimator as the RV proxy. The src/data.py module implements both paths behind a common interface returning [date, rv, log_rv, ret], plus cleaning, split labelling, and a summary-stats helper. The EDA notebook notebooks/01_eda.ipynb was executed end to end and saved two report figures (log-RV time series and SPX ACF/PACF) in SVG, PDF, and PNG, along with a summary-stats CSV. Every series passes the plan's persistence sanity check (lag-1 autocorrelation inside [0.5, 0.8]).

---

## 2. Decisions made

**Decision:** Switch from Oxford-Man to yfinance + Garman-Klass for all four assets.
**Alternatives:** Hunt for a Kaggle/Zenodo mirror of the original 5-minute realized library.
**Reason:** The official source and its mirror both 404; the fallback works, passes the sanity checks, and a same-day deadline does not allow a data-archaeology detour (user confirmed at DECISION POINT 1).

**Decision:** Keep z>8 outliers rather than dropping them, only flooring non-positive Garman-Klass values.
**Alternatives:** Winsorize or drop extreme observations.
**Reason:** The 2008 and 2020 spikes are real volatility events the models must learn, not data errors; the plan explicitly says inspect, do not blindly drop.

---

## 3. Results

### Quantitative

Summary statistics on daily log-RV (full sample):

| Asset | n    | Start      | End        | Mean   | Std   | Skew  | Kurt  | Lag-1 AC |
|-------|------|------------|------------|--------|-------|-------|-------|----------|
| SPX   | 6657 | 2000-01-03 | 2026-06-23 | -10.198| 1.203 | 0.239 | 0.199 | 0.652    |
| FTSE  | 6686 | 2000-01-04 | 2026-06-23 | -10.033| 1.088 | 0.409 | 0.341 | 0.610    |
| N225  | 6481 | 2000-01-04 | 2026-06-22 | -9.995 | 1.038 | 0.272 | 0.382 | 0.561    |
| DAX   | 6721 | 2000-01-03 | 2026-06-23 | -9.665 | 1.161 | 0.293 | 0.174 | 0.656    |

Split sizes (train / val / test): SPX 4779/505/1255, FTSE 4800/506/1261, N225 4662/483/1223, DAX 4822/505/1274.

### Qualitative

**Surprise:** None. The Oxford-Man failure was anticipated in the Phase 0 risk register, so the fallback was already implemented and the decision point resolved in minutes. Persistence is slightly lower than 5-minute RV would give because the Garman-Klass daily proxy is noisier, but all four assets still sit comfortably inside the sanity band.

---

## 4. Report prose draft

> Because the Oxford-Man Institute Realized Library has been discontinued, we construct a daily realized-variance proxy from index OHLC data retrieved through Yahoo Finance for four major equity indices: the S&P 500, FTSE 100, Nikkei 225, and DAX, spanning January 2000 to June 2026. For each trading day we compute the Garman-Klass range-based variance estimator \cite{garmanklass1980}, which combines the high-low range and the open-close move into an efficient single-day volatility estimate, and we model its natural logarithm to stabilise variance and approximate normality. The sample is split chronologically into training (2000 to 2018), validation (2019 to 2020), and test (2021 to 2025) windows, with the validation window deliberately spanning the COVID-19 shock as an out-of-sample stress test. The log-RV series exhibit the canonical features of financial volatility: pronounced clustering around the 2008 and 2020 crises, mild positive skew, and slowly decaying autocorrelation. The sample autocorrelation function decays hyperbolically while the partial autocorrelation cuts off after roughly two lags, the long-memory signature that motivates Corsi's heterogeneous daily, weekly, and monthly decomposition \cite{corsi2009}.

### Citations needed for this prose

- garmanklass1980: M. B. Garman and M. J. Klass, "On the Estimation of Security Price Volatilities from Historical Data", Journal of Business, 1980.
- corsi2009: F. Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics, 2009.

---

## 5. Figures generated

| File | Purpose | Goes in report section |
|------|---------|------------------------|
| `figures/fig_logrv_series.svg/pdf` | Daily log-RV for all four assets, val window shaded | Data |
| `figures/fig_acf_pacf.svg/pdf` | SPX log-RV ACF/PACF, long-memory signature | Data / Related work |
| `figures/summary_stats.csv` | Summary-stats table source | Data (Table) |

---

## 6. Files created or modified

```
src/data.py (modified, split-column dtype fix; 240 lines)
notebooks/01_eda.ipynb (new, executed with outputs)
data/processed/SPX.parquet, FTSE.parquet, N225.parquet, DAX.parquet (new, gitignored)
figures/fig_logrv_series.{svg,pdf,png} (new)
figures/fig_acf_pacf.{svg,pdf,png} (new)
figures/summary_stats.csv (new)
docs/journal/phase_1_data.md (new)
```

---

## 7. AI tool usage in this phase

**Tool:** Claude Code (claude-opus-4-8)
**Task:** Authored src/data.py (Oxford-Man and yfinance Garman-Klass loaders, cleaning, splits, summary stats) and notebooks/01_eda.ipynb; ran the ingestion and the EDA notebook.
**Verification:** Probed both data sources directly before committing to one; confirmed every asset's lag-1 autocorrelation falls in the plan's [0.5, 0.8] sanity band; visually inspected both rendered figures (clustering at 2008/2020, slow ACF decay with sharp PACF cutoff); checked split-size arithmetic against the date boundaries.

---

## 8. Risk register

**Next phase risk:** The Garman-Klass proxy is noisier than 5-minute RV, so HAR-RV log-RMSE may land at the upper end of the [0.3, 0.6] sanity band; if it exceeds 0.6, revisit the RV computation before neural models (DECISION POINT 2).
**Final report risk:** The data section must state clearly that RV is a daily range-based proxy, not intraday realized variance, to avoid overclaiming.
**Submission risk:** None at this stage.

---

## 9. Lessons

Anticipating the Oxford-Man failure in Phase 0 and implementing the fallback up front turned a potential multi-hour blocker into a five-minute decision. The Garman-Klass proxy trades a little persistence for full reproducibility from a free, always-available source, which is the right call under deadline.

---

## 10. Prep for next phase

1. Implement src/baselines.py: fit_garch (arch) and fit_har_rv (statsmodels OLS on daily/weekly/monthly lag aggregates).
2. Add rmse and qlike to src/evaluate.py (QLIKE on RV space, not log).
3. Log baseline coefficients and metrics to MLflow experiment "baselines" using the SQLite backend, and check HAR-RV coefficients are positive and roughly sum to 1.

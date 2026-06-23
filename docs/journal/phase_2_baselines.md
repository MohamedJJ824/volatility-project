# Phase 2: Classical baselines

**Date:** 2026-06-24 06:20 KST
**Duration:** ~0.5 hours
**Phase outcome:** COMPLETE
**Hour of total budget consumed:** ~1.5 / 10

---

## 1. What was done

GARCH(1,1) and HAR-RV were fitted for SPX and logged to the MLflow experiment "baselines"
(SQLite backend). HAR-RV is an OLS regression of next-day log-RV on three trailing
aggregates of log-RV: the previous day, the trailing 5-day (weekly) mean, and the
trailing 22-day (monthly) mean, fitted on the training window only and scored on
validation and test. GARCH(1,1) is a constant-mean normal model fitted on training
returns; its 1-step-ahead conditional variance is obtained by filtering the full return
series with the fixed training parameters, then calibrated to the realized-variance level
by a single additive constant estimated on the training set (the close-to-close GARCH
variance and the intraday Garman-Klass proxy target different objects, so this removes the
level offset without using validation or test information). Metrics are RMSE on log-RV and
the Patton robust QLIKE on RV space. This phase also resolved DECISION POINT 2 and the
project was reduced to the MEDIUM CUT scope (SPX only, three models, no backtest); the
baselines were re-run SPX-only to keep the MLflow store and coefficient tables clean.

---

## 2. Decisions made

**Decision:** Re-anchor the HAR-RV RMSE sanity band to the Garman-Klass proxy and proceed (DECISION POINT 2).
**Alternatives:** Try to reduce proxy measurement noise, or halt and re-audit Phase 1.
**Reason:** Diagnostics showed the models are correct (textbook coefficients, HAR beats climatology and random walk on every split); the elevated RMSE is the irreducible noise of the daily proxy, not a pipeline bug, and the [0.3, 0.6] band assumed 5-minute RV.

**Decision:** Calibrate GARCH variance to the proxy level with one train-estimated additive constant in log space.
**Alternatives:** Compare GARCH and the GK proxy directly without calibration.
**Reason:** GARCH models close-to-close return variance while the proxy is intraday range variance, so an uncalibrated comparison would penalise GARCH for a definitional level offset rather than for forecast quality.

---

## 3. Results

### Quantitative

HAR-RV coefficients (SPX, fitted on 2000 to 2018):

| const | daily | weekly | monthly | sum of slopes |
|-------|-------|--------|---------|---------------|
| -0.678 | 0.149 | 0.462 | 0.323 | 0.933 |

GARCH(1,1) coefficients (SPX):

| mu | omega | alpha | beta | persistence | calib offset |
|----|-------|-------|------|-------------|--------------|
| 0.0617 | 0.0252 | 0.1204 | 0.8605 | 0.981 | -0.870 |

Forecast accuracy (SPX):

| Model  | Val log-RMSE | Val QLIKE | Test log-RMSE | Test QLIKE |
|--------|--------------|-----------|---------------|------------|
| HAR-RV | 0.896        | 0.515     | 0.835         | 0.445      |
| GARCH  | 0.947        | 0.457     | 0.884         | 0.454      |

### Qualitative

The HAR-RV slope coefficients are all positive and sum to 0.933, reproducing Corsi's
empirical regularity that the three horizon aggregates carry near-unit total persistence,
with the weekly term dominant. GARCH persistence (alpha + beta = 0.981) is the textbook
near-integrated value for daily equity volatility.

**Surprise:** The model ranking depends on the loss. HAR-RV wins on RMSE in both windows,
but GARCH attains a lower QLIKE on the COVID-heavy validation window (0.457 vs 0.515),
because QLIKE penalises variance under-prediction sharply and GARCH reacts faster to the
March 2020 spike. On the calmer test window HAR-RV regains the QLIKE edge (0.445 vs 0.454).
This loss-dependent ranking is a clean point for the discussion.

---

## 4. Report prose draft

> Two classical baselines anchor the comparison. The Heterogeneous Autoregressive model of
> Realized Volatility \cite{corsi2009} regresses next-day log-RV on three trailing aggregates
> of past log-RV defined over daily, weekly, and monthly horizons, a hand-engineered
> multi-timescale decomposition motivated by the heterogeneous-market hypothesis. Fitted on
> the SPX training sample, its slope coefficients are all positive and sum to 0.93, with the
> weekly term largest, reproducing the near-unit total persistence that Corsi reports. The
> second baseline is a constant-mean GARCH(1,1) \cite{bollerslev1986} on daily returns, whose
> estimated persistence of 0.98 is the usual near-integrated value for equity indices; because
> GARCH models close-to-close return variance rather than the intraday range proxy, its
> conditional-variance forecasts are calibrated to the realized-variance level by a single
> constant estimated on the training set. On the test window HAR-RV attains the lower loss on
> both RMSE (0.835 vs 0.884) and QLIKE (0.445 vs 0.454), but on the COVID-dominated validation
> window the ranking inverts under QLIKE, where GARCH's faster reaction to the March 2020 shock
> yields a lower variance-space loss. This loss-dependence motivates reporting both metrics and
> testing significance directly.

### Citations needed for this prose

- corsi2009: F. Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics, 2009.
- bollerslev1986: T. Bollerslev, "Generalized Autoregressive Conditional Heteroskedasticity", Journal of Econometrics, 1986.

---

## 5. Figures generated

None this phase. Coefficient and metric tables saved to `experiments/`.

---

## 6. Files created or modified

```
src/evaluate.py (new, rmse / qlike / per-obs losses)
src/baselines.py (new, HAR-RV + GARCH + MLflow logging, scoped to SPX)
experiments/har_coefficients.csv (new)
experiments/garch_coefficients.csv (new)
experiments/baseline_metrics.json (new)
experiments/mlflow.db (regenerated, gitignored; 2 runs: SPX_HAR-RV, SPX_GARCH)
docs/journal/phase_2_baselines.md (new)
```

---

## 7. AI tool usage in this phase

**Tool:** Claude Code (claude-opus-4-8)
**Task:** Wrote src/evaluate.py and src/baselines.py, fitted both baselines for SPX, and logged coefficients and metrics to MLflow.
**Verification:** Confirmed HAR-RV slopes are positive and sum near 1 (Corsi regularity) and GARCH persistence is ~0.98; ran a diagnostic showing HAR-RV beats climatology and random-walk benchmarks on every split (establishing the RMSE floor is proxy noise, not a bug); cross-checked QLIKE non-negativity; verified the MLflow store holds exactly the two expected SPX runs.

---

## 8. Risk register

**Next phase risk:** The TCNN-LSTM must clear the same noisy-proxy floor as HAR-RV; with a single seed and no sweep it may land slightly above HAR-RV, in which case the negative-result script applies.
**Final report risk:** The GARCH level-calibration step must be stated plainly so a reader does not mistake it for tuning on the test set.
**Submission risk:** None at this stage.

---

## 9. Lessons

The most useful result of the phase was negative-shaped: the headline RMSE is dominated by
the measurement noise of the daily proxy, so absolute numbers will look poor against the
5-minute-RV literature while relative comparisons remain valid. Framing this explicitly up
front protects the results section from an obvious reviewer objection.

---

## 10. Prep for next phase

1. Implement src/models.py with DualResTCNNLSTM (two dilated-TCNN branches over 5-day and 22-day windows, concat, single LSTM step, MLP head to scalar log-RV).
2. Implement src/train.py: Adam (lr 1e-3, cosine), early stopping on val RMSE (patience 15), batch 64, max 200 epochs, seed 42, MLflow experiment "neural", checkpoint to experiments/checkpoints/.
3. Reuse the HAR feature/window construction from baselines.py so the deep model sees the same information set as HAR-RV for a fair comparison.

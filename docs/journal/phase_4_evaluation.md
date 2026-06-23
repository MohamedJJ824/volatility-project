# Phase 4: Evaluation

**Date:** 2026-06-24 07:20 KST
**Duration:** ~0.75 hours
**Phase outcome:** COMPLETE
**Hour of total budget consumed:** ~2.75 / 10

---

## 1. What was done

The three models (GARCH, HAR-RV, TCNN-LSTM) were compared statistically on the SPX test set
(1255 days, 2021 to 2026) using RMSE on log-RV and the Patton robust QLIKE on RV, with pairwise
Diebold-Mariano tests (Harvey-Leybourne-Newbold corrected) on both loss functions. Four report
figures were produced: the headline HAR-RV / TCNN-LSTM analogy diagram, the predicted-vs-actual
series, the metrics table, and a robustness/ablation appendix figure. Because the headline result
rested on a single seed, two evaluation-driven experiments were added beyond the MEDIUM CUT: a
five-seed robustness sweep and a short-only / long-only branch ablation. All runs are logged to
the MLflow "neural" experiment; final metrics and DM statistics are saved to
experiments/final_results.json, with robustness.csv and ablation.csv alongside.

---

## 2. Decisions made

**Decision:** Add a five-seed robustness sweep and a branch ablation despite both being on the cut list.
**Alternatives:** Report the single-seed result as-is.
**Reason:** The single-seed model is DM-significant over HAR-RV, but responsible reporting of a small effect requires showing it is not a seed artefact and that the dual-resolution structure does the work; both checks are seconds of compute.

**Decision:** Report the RMSE improvement as the headline and explicitly temper the QLIKE claim.
**Alternatives:** Lead with the seed-42 QLIKE win (0.424 vs 0.445, DM p = 0.030).
**Reason:** The robustness sweep shows the QLIKE advantage is seed-specific (1 of 5 seeds beats HAR-RV on QLIKE) while the RMSE advantage holds for all five seeds, so honesty requires foregrounding the robust result.

---

## 3. Results

### Quantitative

SPX test set (n = 1255):

| Model     | Test log-RMSE | Test QLIKE |
|-----------|---------------|------------|
| GARCH     | 0.884         | 0.454      |
| HAR-RV    | 0.835         | 0.445      |
| TCNN-LSTM | **0.823**     | 0.424 (seed 42) |

Diebold-Mariano (squared error on log-RV; negative statistic favours the first model):

| Pair | DM stat | p | Verdict |
|------|---------|---|---------|
| GARCH vs HAR-RV    | +4.71 | 0.000 | HAR-RV better, significant |
| GARCH vs TCNN-LSTM | +5.38 | 0.000 | TCNN-LSTM better, significant |
| HAR-RV vs TCNN-LSTM| +2.49 | 0.013 | TCNN-LSTM better, significant |

Diebold-Mariano (QLIKE on RV): HAR-RV vs TCNN-LSTM p = 0.030 (seed 42); GARCH vs others not significant.

Robustness (5 seeds, dual model): test log-RMSE mean 0.824, std 0.002, range [0.822, 0.827].
Seeds beating HAR-RV: 5/5 on RMSE, 1/5 on QLIKE.

Ablation (seed 42): dual 0.823 < short-only 0.827 < long-only 0.830 (test log-RMSE).

### Qualitative

**Surprise:** The robustness sweep changed the QLIKE story. The seed-42 QLIKE win is real for that
model and DM-significant, but it does not survive reseeding: across five seeds the TCNN-LSTM QLIKE
clusters around HAR-RV's 0.445. The squared-error improvement, by contrast, is remarkably stable
(std 0.002) and holds for every seed. The ablation confirms the dual-resolution design contributes:
removing either branch raises test RMSE, and the short branch alone is stronger than the long branch
alone, consistent with volatility being most predictable from recent history while the longer horizon
still adds a little. All effects are small in absolute terms, which is the expected consequence of the
noisy daily Garman-Klass proxy.

---

## 4. Report prose draft (Results, ~1 page)

> On the SPX test set the dual-resolution TCNN-LSTM attains the lowest squared-error loss, with a
> test log-RMSE of 0.823 against 0.835 for HAR-RV and 0.884 for GARCH. Diebold-Mariano tests confirm
> the ordering is statistically meaningful under squared-error loss: the TCNN-LSTM improves on HAR-RV
> (DM = 2.49, p = 0.013) and on GARCH (DM = 5.38, p < 0.001), and HAR-RV in turn improves on GARCH
> (DM = 4.71, p < 0.001). The absolute level of all three RMSE values is high relative to the
> realized-volatility literature because the target is a daily range-based proxy rather than a
> 5-minute realized measure; this measurement noise is irreducible and bounds the achievable RMSE for
> every model equally, so the relative comparison is the informative one. Under the QLIKE loss the
> picture is more nuanced. The selected TCNN-LSTM improves significantly on HAR-RV (p = 0.030), but a
> five-seed robustness sweep shows this QLIKE advantage is seed-specific: only one of five seeds beats
> HAR-RV on QLIKE, whereas all five beat it on RMSE, with a test log-RMSE of 0.824 plus or minus 0.002.
> We therefore report the squared-error improvement as the robust finding and treat the QLIKE result
> as indicative rather than conclusive. A branch ablation isolates the contribution of the dual-resolution
> design: removing the short branch raises test RMSE to 0.827 and removing the long branch to 0.830,
> so the combined model is best and both timescales carry signal, with the recent-history branch
> dominant. Taken together, the architecture transfer succeeds in a measurable but modest way: replacing
> Corsi's fixed linear aggregation of daily, weekly, and monthly volatility with learned convolutional
> branches over the same information set yields a small, robust, and statistically significant reduction
> in squared forecast error, while leaving variance-space loss essentially unchanged.

### Citations needed for this prose

- corsi2009: F. Corsi, 2009 (as before).
- dieboldmariano1995: F. X. Diebold and R. S. Mariano, "Comparing Predictive Accuracy", Journal of Business and Economic Statistics, 1995.
- patton2011: A. J. Patton, "Volatility forecast comparison using imperfect volatility proxies", Journal of Econometrics, 2011.

---

## 5. Figures generated

| File | Purpose | Goes in report section |
|------|---------|------------------------|
| `figures/fig_har_analogy.svg/pdf` | Headline: HAR-RV lags beside the short/long TCN branches | Related work / Method |
| `figures/fig_predictions.svg/pdf` | Actual vs predicted log-RV, SPX test window, all models | Results |
| `figures/fig_metrics_table.svg/pdf` | Test RMSE/QLIKE + DM p-values | Results |
| `figures/fig_robustness_ablation.svg/pdf` | 5-seed robustness and branch ablation | Appendix |

---

## 6. Files created or modified

```
src/evaluate.py (extended: diebold_mariano, dm_from_losses, evaluation main)
src/models.py (added branches='both'|'short'|'long')
src/train.py (refactored into train_once for seed/ablation sweeps)
src/experiments_extra.py (new: robustness + ablation driver)
experiments/final_results.json (new: metrics + DM)
experiments/robustness.csv, experiments/ablation.csv (new)
figures/fig_har_analogy.*, fig_predictions.*, fig_metrics_table.*, fig_robustness_ablation.* (new)
figures/metrics_table.csv (new)
docs/journal/phase_4_evaluation.md (new)
```

---

## 7. AI tool usage in this phase

**Tool:** Claude Code (claude-opus-4-8)
**Task:** Implemented the Diebold-Mariano test and evaluation driver, generated all four figures, and built and ran the robustness and ablation experiments.
**Verification:** Confirmed DM statistic signs against the metric ordering (negative favours the lower-loss model) and that QLIKE losses are non-negative; cross-checked that robustness seeds reuse the identical pipeline and that all five land within a 0.005 RMSE band; verified the ablation models differ only in active branches; inspected every rendered figure.

---

## 8. Risk register

**Next phase risk:** The report must present the QLIKE tempering clearly so the contribution is not overstated; the headline claim is the robust RMSE improvement, not a blanket win.
**Final report risk:** Page budget is 5 to 6 pages; with four figures plus a table, figure sizing must be controlled to avoid overflow.
**Submission risk:** None; results are final and saved.

---

## 9. Lessons

Spending five minutes on a robustness sweep converted a fragile single-seed claim into a defensible
one and simultaneously caught an overstatement (the QLIKE win). The cut list was right to defer these
as optional, but they were cheap enough that the evaluation phase was the correct place to add them
back once the headline result proved significant.

---

## 10. Prep for next phase

1. Assemble the report from the journal prose drafts (Phases 1 to 4); the Results draft above is near-final.
2. Build the LaTeX skeleton (abstract, intro/motivation, related work + analogy figure, method, data, results, discussion, conclusion, appendix with hyperparameters, robustness/ablation, and the AI tool statement).
3. Keep the discussion honest about the proxy-noise RMSE floor and the seed-dependent QLIKE result.

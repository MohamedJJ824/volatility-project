# Phase 3: Neural model

**Date:** 2026-06-24 06:45 KST
**Duration:** ~0.5 hours
**Phase outcome:** COMPLETE
**Hour of total budget consumed:** ~2.0 / 10

---

## 1. What was done

The dual-resolution TCNN-LSTM was ported from the published ADDIM Human Activity Recognition
architecture and trained on SPX log-RV under the MEDIUM CUT (one asset, single seed 42). The
model has a short branch over a 5-day window and a long branch over a 22-day window, each a
stack of dilated 1D convolutions (kernel 3, dilations 1, 2, 4) followed by global average
pooling; the two pooled feature vectors are concatenated and passed as a length-1 sequence
through a single LSTM step and an MLP head that emits a scalar log-RV forecast. Inputs were
standardised with training-set statistics only, and the model was trained with Adam (lr 1e-3,
cosine schedule), batch size 64, and early stopping on validation RMSE (patience 15). Training
converged quickly: the best validation epoch was 27 and training stopped at epoch 42. The run,
its per-epoch curves, the best checkpoint, and val/test predictions are all logged to the
MLflow "neural" experiment and saved under experiments/. The model has 89,153 parameters and
trains in seconds on CPU.

---

## 2. Decisions made

**Decision:** Standardise inputs and target with training-set mean and standard deviation, inverting at evaluation.
**Alternatives:** Train directly on raw log-RV around -10.
**Reason:** Centred, unit-scale inputs train far more stably; using train-only statistics keeps the transform leakage-free.

**Decision:** Implement the fusion exactly as specified, as a length-1 LSTM step over the concatenated pooled features.
**Alternatives:** Feed the full convolutional time series into the LSTM.
**Reason:** Fidelity to the published architecture is the point of the transfer; the branches already pool temporally, and the LSTM step acts as the gated fusion described in the original model.

---

## 3. Results

### Quantitative

All three models on SPX (log-RMSE and Patton QLIKE):

| Model     | Val log-RMSE | Val QLIKE | Test log-RMSE | Test QLIKE |
|-----------|--------------|-----------|---------------|------------|
| GARCH     | 0.947        | 0.457     | 0.884         | 0.454      |
| HAR-RV    | 0.896        | 0.515     | 0.835         | 0.445      |
| TCNN-LSTM | **0.879**    | 0.471     | **0.823**     | **0.424**  |

Best validation epoch 27; early stop at 42; 89,153 parameters.

### Qualitative

**Surprise:** The transferred architecture actually beats both baselines on RMSE in both
windows and on test QLIKE, rather than merely matching them. The margins are small (about two
percent of RMSE), which is consistent with the irreducible measurement noise of the daily
Garman-Klass proxy; whether the gap is statistically meaningful is exactly what the
Diebold-Mariano tests in Phase 4 will decide. On validation QLIKE the fast-reacting GARCH still
edges ahead during the COVID window, so the deep model's advantage is clearest on the calmer
test period.

---

## 4. Report prose draft

> The proposed model transfers a dual-resolution TCNN-LSTM, originally published for human
> activity recognition \cite{addim2026}, to volatility forecasting. The architectural prior is
> the same one that makes HAR-RV effective: volatility carries information at multiple
> timescales, and a forecaster benefits from processing short and long histories on separate
> pathways before combining them. Where HAR-RV \cite{corsi2009} hand-engineers this as a linear
> combination of the previous day, the trailing week, and the trailing month, the TCNN-LSTM
> learns it: a short branch consumes a five-day window and a long branch a twenty-two day window,
> each a stack of dilated temporal convolutions whose growing receptive field aggregates the
> window at increasing scales. The two branches are globally pooled, concatenated, and fused
> through a single recurrent step before an MLP head emits the next-day log-RV. The short and
> long windows are chosen to span the same information set HAR-RV uses, so the comparison
> isolates the value of replacing a fixed linear aggregation with a learned nonlinear one. The
> model is deliberately small, 89k parameters, and is trained with Adam under a cosine schedule
> with early stopping on validation RMSE, matching the reproducibility constraints of the study.

### Citations needed for this prose

- corsi2009: F. Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics, 2009.
- addim2026: M. Diallo et al., dual-resolution TCNN-LSTM for Human Activity Recognition, CEA, 2026.

---

## 5. Figures generated

| File | Purpose | Goes in report section |
|------|---------|------------------------|
| `figures/fig_training_curves.svg/pdf` | Train MSE and val log-RMSE vs epoch, best epoch and HAR-RV line marked | Method / Results |

---

## 6. Files created or modified

```
src/models.py (new, DualResTCNNLSTM + TCNBranch)
src/train.py (new, windows, standardisation, training loop, MLflow, checkpoint)
experiments/checkpoints/tcnnlstm_seed42.pt (new, gitignored)
experiments/neural_history.csv (new)
experiments/neural_predictions.csv (new, val+test predictions for Phase 4)
figures/fig_training_curves.{svg,pdf,png} (new)
docs/journal/phase_3_neural.md (new)
```

---

## 7. AI tool usage in this phase

**Tool:** Claude Code (claude-opus-4-8)
**Task:** Implemented src/models.py and src/train.py, trained the TCNN-LSTM on SPX, and produced the training-curves figure.
**Verification:** Confirmed the parameter count and that padding=dilation preserves window length; checked the loss decreased monotonically and validation RMSE tracked sensibly with early stopping selecting epoch 27; verified the reported val/test metrics against the same evaluate.py functions used for the baselines (identical information set and target), and inspected the rendered training-curve figure.

---

## 8. Risk register

**Next phase risk:** The TCNN-LSTM's edge over HAR-RV is small, so the Diebold-Mariano test may report it as not statistically significant, in which case the negative-result framing applies.
**Final report risk:** The Method section must make the shared multi-timescale prior explicit with the headline analogy figure, or the architectural-transfer story will not land.
**Submission risk:** None at this stage.

---

## 9. Lessons

A faithful, deliberately small port of the published architecture was enough to edge past both
classical baselines, which strengthens the transfer narrative without overfitting risk. The
single-seed scope means the result should be reported as a point estimate with significance
testing rather than as a seed-averaged claim.

---

## 10. Prep for next phase

1. Extend src/evaluate.py with diebold_mariano(e1, e2, h=1, loss) and run pairwise DM tests across GARCH, HAR-RV, TCNN-LSTM on the test set.
2. Draw the headline fig_har_analogy (HAR daily/weekly/monthly lags beside the short/long TCNN branches) and fig_predictions (predicted vs actual log-RV, all three models, SPX test window).
3. Assemble the RMSE/QLIKE metrics table and write the full Results prose.

# Phase N: <Name>

**Date:** YYYY-MM-DD HH:MM KST
**Duration:** <actual hours spent>
**Phase outcome:** COMPLETE | PARTIAL | FAILED
**Hour of total budget consumed:** <X / 36>

---

## 1. What was done

One paragraph, plain prose. What got built or computed in this phase. No bullet points.

---

## 2. Decisions made

For each significant decision:

**Decision:** <what was chosen>
**Alternatives:** <what else was considered>
**Reason:** <one sentence>

Repeat for each. Skip decisions that were forced by the plan; only record genuine choice points.

---

## 3. Results

### Quantitative

Tables go here. Example:

| Asset | Model | Val log-RMSE | Val QLIKE |
|-------|-------|--------------|-----------|
| SPX   | HAR-RV | 0.412       | 0.187     |

### Qualitative

Prose for surprises, anomalies, or things that don't fit a table. Flag surprises with **Surprise:** so they're scannable.

---

## 4. Report prose draft

One to three paragraphs of LaTeX-ready prose. This text should be paste-able into the final report with minimal edits. No em dashes. Concise and technical. Use `\cite{}` placeholders.

Example opener:
> The Heterogeneous Autoregressive model of Realized Volatility \cite{corsi2009} forecasts next-day RV from three linearly combined lag aggregates: the previous day, the previous week, and the previous month. This decomposition reflects the empirical observation that volatility exhibits heterogeneous persistence across investor horizons...

### Citations needed for this prose

- corsi2009: F. Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics, 2009.
- <add more as needed>

---

## 5. Figures generated

| File | Purpose | Goes in report section |
|------|---------|------------------------|
| `figures/fig_X.svg` | Description | Section name |

---

## 6. Files created or modified

```
src/data.py (new, 84 lines)
notebooks/01_eda.ipynb (new)
data/processed/SPX.parquet (new)
```

---

## 7. AI tool usage in this phase

For each tool interaction:

**Tool:** Claude Code (claude-opus-4-7)
**Task:** Generated initial draft of `src/data.py`.
**Verification:** Read function by function, ran unit test on synthetic data, manually compared output against a known SPX RV value for 2020-03-16.

Repeat per interaction.

---

## 8. Risk register

**Next phase risk:** <one sentence>
**Final report risk:** <one sentence>
**Submission risk:** <one sentence>

---

## 9. Lessons (especially if PARTIAL or FAILED)

What did this phase teach that the plan didn't anticipate. One paragraph.

---

## 10. Prep for next phase

Up to three concrete actions before the next phase starts:

1.
2.
3.

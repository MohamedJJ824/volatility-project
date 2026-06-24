# Phase 5: Report

**Date:** 2026-06-24 09:05 KST
**Duration:** ~0.75 hours
**Phase outcome:** COMPLETE
**Hour of total budget consumed:** ~3.5 / 10

---

## 1. What was done

The full report was written in LaTeX (clean article class) and compiled to a 7-page PDF
(report/main.pdf): roughly 5 pages of body across the six required rubric sections plus a
references page and a short appendix. Content was assembled from the Phase 1 to 4 journal prose
drafts with light editing, and grounded in the two source documents the user supplied: the course
brief (confirming the six sections, the 4 to 8 page guidance, and the AI-tool disclosure
requirement) and the CEA 2026 HAR paper (supplying the real citation and architecture). The
bibliography (refs.bib) holds eight references with verified details. Five figures are embedded
(analogy, SPX series, ACF/PACF, predictions, robustness/ablation) plus the training curve in the
appendix, and two tables (accuracy, Diebold-Mariano). The AI-tool usage statement required by the
rubric is included in the appendix.

---

## 2. Decisions made

**Decision:** Frame the CEA 2026 model as co-authored prior work ("we co-proposed", \cite{warnants2026}), not as sole-authored.
**Alternatives:** Keep the plan's "I published" phrasing.
**Reason:** The paper lists the author as second of five (Warnants, Diallo, Ortiz, Garrigos, Vera); honesty and the rubric's integrity expectations require accurate attribution.

**Decision:** Lead the results with the robust squared-error improvement and explicitly temper the QLIKE claim.
**Alternatives:** Headline the seed-42 QLIKE win.
**Reason:** The five-seed sweep shows the QLIKE advantage is not robust; the rubric rewards awareness of limitations.

**Decision:** Consolidate the report into a single main.tex rather than split section files.
**Alternatives:** Use report/sections/ as in the original plan tree.
**Reason:** One file compiles more reliably under deadline; the section split added risk without benefit.

---

## 3. Results

### Quantitative

| Artifact | Value |
|----------|-------|
| PDF pages (total) | 7 |
| Body pages (excl. refs + appendix) | ~5 |
| Rubric sections present | 6/6 |
| Figures embedded | 5 body + 1 appendix |
| Tables | 2 (accuracy, DM) |
| References | 8, all resolved |
| Compile | clean (no undefined refs, no large overfull boxes) |

### Qualitative

The report maps cleanly onto the rubric's evaluation criteria: it foregrounds the logical chain
from the financial problem (volatility forecasting) through the inductive bias (multi-timescale)
to the method (architecture transfer), the implementation, and a careful evaluation with explicit
limitations. The honest QLIKE tempering and the proxy-noise discussion directly serve the
"awareness of limitations and risks" criterion.

---

## 4. Report prose draft

Not applicable; this phase produced the report itself rather than a draft for a later phase.

---

## 5. Figures generated

| File | Used in |
|------|---------|
| `fig_har_analogy.pdf` | Related Work (headline) |
| `fig_spx_logrv.pdf`, `fig_acf_pacf.pdf` | Data |
| `fig_predictions.pdf`, `fig_robustness_ablation.pdf` | Results |
| `fig_training_curves.pdf` | Appendix |

---

## 6. Files created or modified

```
report/main.tex (new, full report)
report/refs.bib (new, 8 references)
report/main.pdf (new, 7 pages, compiled deliverable)
figures/fig_spx_logrv.{svg,pdf,png} (new, SPX-only series for Data section)
docs/journal/phase_5_report.md (new)
```

---

## 7. AI tool usage in this phase

**Tool:** Claude Code (claude-opus-4-8)
**Task:** Wrote report/main.tex and report/refs.bib, generated the SPX series figure, and compiled the PDF.
**Verification:** Extracted the course rubric and the CEA 2026 citation/architecture directly from the supplied PDFs (rendering the image-based HAR paper to confirm authorship and the 2-minute/2-hour dual-resolution design); confirmed the compile is clean with no undefined references or missing figures; visually inspected the rendered pages for layout and figure placement; cross-checked every reported number against experiments/final_results.json and the journals.

---

## 8. Risk register

**Next phase risk:** The repository is still local; Phase 6 must push it, make it public, and replace the placeholder repo URL in the report appendix before submission.
**Final report risk:** None outstanding; the PDF compiles and matches the rubric.
**Submission risk:** Deadline is today June 24 24:00 KST; submit with buffer and verify the uploaded PDF opens.

---

## 9. Lessons

Having the four phase journals already written turned report assembly into editing rather than
writing, which is exactly what the journaling discipline was meant to buy. The one real dependency,
the author's own prior paper, could not be fabricated and rightly required the source document.

---

## 10. Prep for next phase

1. Initialize a public GitHub repository and push; replace the placeholder URL in report/main.tex appendix, recompile.
2. Write the public README leading with the analogy figure and the results table.
3. Tag a release and submit the PDF on the course platform before 24:00 KST.

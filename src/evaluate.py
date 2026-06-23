"""Evaluation metrics for the IE412 volatility project.

Statistical losses used throughout:
  - rmse:  computed on log realized variance (the modelling target).
  - qlike: the Patton (2011) robust QLIKE loss, computed on RV (variance) space.
           QLIKE(s, h) = s/h - log(s/h) - 1, where s is the realized proxy and h
           the forecast. It is non-negative and zero only at h == s, which is why
           the Phase 4 sanity check expects no negative QLIKE.

The Diebold-Mariano test is added here in Phase 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y, yhat) -> float:
    """Root mean squared error on log-RV (or any common scale)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))


def qlike(rv, rv_hat) -> float:
    """Patton robust QLIKE on RV (variance) space. Inputs must be strictly positive."""
    rv = np.asarray(rv, dtype=float)
    rv_hat = np.asarray(rv_hat, dtype=float)
    m = np.isfinite(rv) & np.isfinite(rv_hat) & (rv > 0) & (rv_hat > 0)
    r = rv[m] / rv_hat[m]
    return float(np.mean(r - np.log(r) - 1.0))


def squared_errors(y, yhat) -> np.ndarray:
    """Per-observation squared errors on log-RV (for the Diebold-Mariano test)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return (y - yhat) ** 2


def qlike_errors(rv, rv_hat) -> np.ndarray:
    """Per-observation QLIKE losses on RV space (for the Diebold-Mariano test)."""
    rv = np.asarray(rv, dtype=float)
    rv_hat = np.asarray(rv_hat, dtype=float)
    r = rv / rv_hat
    return r - np.log(r) - 1.0


# --- Diebold-Mariano test ---------------------------------------------------

def dm_from_losses(loss1, loss2, h: int = 1):
    """Diebold-Mariano test from per-observation losses, with the Harvey-Leybourne-Newbold
    small-sample correction. Returns (statistic, p_value). A negative statistic means model 1
    has the lower loss (is more accurate); the p-value is two-sided.
    """
    from scipy import stats

    d = np.asarray(loss1, dtype=float) - np.asarray(loss2, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    var = np.mean(dc ** 2)
    for k in range(1, h):  # h-1 autocovariances for h-step forecasts
        var += 2.0 * np.mean(dc[k:] * dc[:-k])
    var_dbar = var / n
    dm = dbar / np.sqrt(var_dbar)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    p = 2.0 * stats.t.cdf(-abs(dm_hln), df=n - 1)
    return float(dm_hln), float(p)


def diebold_mariano(e1, e2, h: int = 1, loss: str = "se"):
    """Diebold-Mariano test from forecast errors e = y - yhat.

    loss = 'se' uses squared-error loss; loss = 'ae' uses absolute-error loss.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if loss == "se":
        l1, l2 = e1 ** 2, e2 ** 2
    elif loss in ("ae", "ad"):
        l1, l2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError(f"unknown loss {loss!r}")
    return dm_from_losses(l1, l2, h=h)


def _load_test_predictions():
    """Assemble aligned SPX test-set predictions for all three models (log-RV space)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import baselines as B

    root = Path(__file__).resolve().parent.parent
    base = B.run_asset("SPX")["predictions"]  # date, split, y, har_log, garch_log
    neural = pd.read_csv(root / "experiments" / "neural_predictions.csv", parse_dates=["date"])
    base["date"] = pd.to_datetime(base["date"])
    df = base.merge(neural[["date", "tcnn_pred_log"]], on="date", how="inner")
    return df[df["split"] == "test"].reset_index(drop=True)


def main():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    df = _load_test_predictions()
    y = df["y"].to_numpy()
    rv = np.exp(y)
    preds = {
        "GARCH": df["garch_log"].to_numpy(),
        "HAR-RV": df["har_log"].to_numpy(),
        "TCNN-LSTM": df["tcnn_pred_log"].to_numpy(),
    }

    metrics = {m: {"test_log_rmse": rmse(y, p), "test_qlike": qlike(rv, np.exp(p))}
               for m, p in preds.items()}

    # Pairwise DM on squared-error (log-RV) and QLIKE (RV) losses.
    names = list(preds)
    dm_se, dm_ql = {}, {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            stat_se, p_se = diebold_mariano(y - preds[a], y - preds[b], h=1, loss="se")
            stat_ql, p_ql = dm_from_losses(qlike_errors(rv, np.exp(preds[a])),
                                           qlike_errors(rv, np.exp(preds[b])), h=1)
            dm_se[f"{a} vs {b}"] = {"stat": stat_se, "p": p_se}
            dm_ql[f"{a} vs {b}"] = {"stat": stat_ql, "p": p_ql}

    out = {"n_test": int(len(df)), "metrics": metrics,
           "dm_squared_error": dm_se, "dm_qlike": dm_ql}
    with open(root / "experiments" / "final_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"=== SPX test set (n={len(df)}) ===")
    for m, v in metrics.items():
        print(f"{m:10s} log-RMSE={v['test_log_rmse']:.3f}  QLIKE={v['test_qlike']:.3f}")
    print("\n=== Diebold-Mariano (squared error on log-RV; neg stat = first model better) ===")
    for k, v in dm_se.items():
        sig = "significant" if v["p"] < 0.05 else "not significant"
        print(f"{k:24s} stat={v['stat']:+.2f}  p={v['p']:.3f}  ({sig})")
    print("\n=== Diebold-Mariano (QLIKE on RV) ===")
    for k, v in dm_ql.items():
        sig = "significant" if v["p"] < 0.05 else "not significant"
        print(f"{k:24s} stat={v['stat']:+.2f}  p={v['p']:.3f}  ({sig})")


if __name__ == "__main__":
    main()

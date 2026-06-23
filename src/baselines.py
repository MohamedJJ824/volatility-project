"""Classical volatility baselines: GARCH(1,1) and HAR-RV.

HAR-RV (Corsi 2009) regresses next-day log-RV on three lag aggregates of log-RV:
the previous day, the trailing 5-day (weekly) mean, and the trailing 22-day (monthly)
mean. GARCH(1,1) models the conditional variance of daily returns.

The two models target different objects (GARCH: close-to-close return variance;
HAR-RV: the intraday Garman-Klass realized-variance proxy), so GARCH forecasts are
calibrated by a single additive constant in log space, estimated on the training set
only, to remove the systematic level offset between close-to-close variance and the
intraday proxy. This makes the level-free dynamics comparable; no validation or test
information enters the calibration.

Run as a script to fit both models per asset, log everything to the MLflow
experiment "baselines", and write coefficient/result tables under experiments/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import evaluate as E

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EXP_DIR = ROOT / "experiments"
TRACKING_URI = "sqlite:///experiments/mlflow.db"
HAR_FEATURES = ["daily", "weekly", "monthly"]

# Project scope (MEDIUM CUT): SPX only.
ASSETS = ["SPX"]


# --- HAR-RV -----------------------------------------------------------------

def har_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build HAR features aligned to the target day.

    For target log_rv at day t (predicted using information through t-1):
      daily   = log_rv[t-1]
      weekly  = mean(log_rv[t-5 : t-1])   (trailing 5 days)
      monthly = mean(log_rv[t-22 : t-1])  (trailing 22 days)
    """
    s = pd.Series(df["log_rv"].to_numpy())
    feat = pd.DataFrame(
        {
            "date": df["date"].to_numpy(),
            "y": s.to_numpy(),
            "daily": s.shift(1).to_numpy(),
            "weekly": s.rolling(5).mean().shift(1).to_numpy(),
            "monthly": s.rolling(22).mean().shift(1).to_numpy(),
            "split": df["split"].to_numpy(),
        }
    )
    return feat.dropna(subset=HAR_FEATURES).reset_index(drop=True)


def fit_har_rv(feat_train: pd.DataFrame):
    """OLS of next-day log-RV on the three HAR lag aggregates. Returns statsmodels result."""
    X = sm.add_constant(feat_train[HAR_FEATURES])
    return sm.OLS(feat_train["y"].astype(float), X).fit()


def har_predict(res, feat: pd.DataFrame) -> np.ndarray:
    X = sm.add_constant(feat[HAR_FEATURES], has_constant="add")
    return res.predict(X).to_numpy()


# --- GARCH(1,1) -------------------------------------------------------------

def fit_garch(train_returns: pd.Series):
    """Fit a constant-mean GARCH(1,1) with normal errors on training returns.

    Returns are scaled to percent for numerical stability (arch convention).
    """
    r = pd.Series(train_returns).dropna().astype(float) * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        return am.fit(disp="off")


def garch_conditional_variance(params, full_returns: pd.Series) -> pd.Series:
    """1-step-ahead conditional variance over the full sample, using fixed train params.

    The GARCH(1,1) conditional variance at day t is the 1-step-ahead forecast formed at
    t-1 from past returns, so filtering the full return series with fixed parameters
    yields a leakage-free forecast for every val/test day. Returned in return^2 units.
    """
    r = pd.Series(full_returns).astype(float) * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(r.dropna(), mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        fixed = am.fix(params)
    cv = pd.Series(fixed.conditional_volatility ** 2, index=r.dropna().index)
    return cv / (100.0 ** 2)  # back to return^2 variance


# --- Per-asset driver -------------------------------------------------------

def run_asset(ticker: str) -> dict:
    """Fit both baselines for one asset and return metrics + coefficients + predictions."""
    df = pd.read_parquet(PROC / f"{ticker}.parquet").sort_values("date").reset_index(drop=True)

    # HAR-RV
    feat = har_features(df)
    tr = feat[feat["split"] == "train"]
    har_res = fit_har_rv(tr)
    feat = feat.assign(har_log=har_predict(har_res, feat))

    # GARCH: filter full returns with train-estimated params; g_cv is indexed by date.
    returns_by_date = df.set_index("date")["ret"]
    g_res = fit_garch(returns_by_date)
    g_cv = garch_conditional_variance(g_res.params, returns_by_date)
    garch_var = pd.DataFrame({"date": g_cv.index, "garch_var": g_cv.to_numpy()})
    feat = feat.merge(garch_var, on="date", how="left")

    # Calibrate GARCH log-variance to the proxy level on TRAIN only.
    tr_mask = feat["split"] == "train"
    valid = tr_mask & feat["garch_var"].notna() & (feat["garch_var"] > 0)
    offset = float(feat.loc[valid, "y"].mean() - np.log(feat.loc[valid, "garch_var"]).mean())
    feat["garch_log"] = np.log(feat["garch_var"]) + offset

    # Metrics per split.
    metrics = {}
    for split in ("val", "test"):
        s = feat[feat["split"] == split].dropna(subset=["y", "har_log", "garch_log"])
        y, rv = s["y"].to_numpy(), np.exp(s["y"].to_numpy())
        metrics[f"HAR-RV/{split}/log_rmse"] = E.rmse(y, s["har_log"])
        metrics[f"HAR-RV/{split}/qlike"] = E.qlike(rv, np.exp(s["har_log"]))
        metrics[f"GARCH/{split}/log_rmse"] = E.rmse(y, s["garch_log"])
        metrics[f"GARCH/{split}/qlike"] = E.qlike(rv, np.exp(s["garch_log"]))

    har_coef = {k: float(v) for k, v in har_res.params.items()}
    garch_coef = {k: float(v) for k, v in g_res.params.items()}
    garch_coef["calib_offset"] = offset

    return {
        "ticker": ticker,
        "metrics": metrics,
        "har_coef": har_coef,
        "garch_coef": garch_coef,
        "predictions": feat[["date", "split", "y", "har_log", "garch_log"]],
    }


def main():
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("baselines")

    all_metrics, har_rows, garch_rows = {}, [], []
    for ticker in ASSETS:
        print(f"[baselines] fitting {ticker} ...", flush=True)
        out = run_asset(ticker)
        all_metrics[ticker] = out["metrics"]

        for model, coef in (("HAR-RV", out["har_coef"]), ("GARCH", out["garch_coef"])):
            with mlflow.start_run(run_name=f"{ticker}_{model}"):
                mlflow.log_params({"asset": ticker, "model": model})
                safe = {f"coef_{k.replace('[', '_').replace(']', '')}": round(v, 6)
                        for k, v in coef.items()}
                mlflow.log_params(safe)
                mlflow.log_metrics(
                    {k.replace("/", "_"): v for k, v in out["metrics"].items() if k.startswith(model)}
                )

        c = out["har_coef"]
        har_rows.append(
            {"asset": ticker, "const": c.get("const"), "daily": c.get("daily"),
             "weekly": c.get("weekly"), "monthly": c.get("monthly"),
             "sum_betas": c.get("daily", 0) + c.get("weekly", 0) + c.get("monthly", 0)}
        )
        g = out["garch_coef"]
        garch_rows.append(
            {"asset": ticker, "mu": g.get("mu"), "omega": g.get("omega"),
             "alpha": g.get("alpha[1]"), "beta": g.get("beta[1]"),
             "persistence": g.get("alpha[1]", 0) + g.get("beta[1]", 0),
             "calib_offset": g.get("calib_offset")}
        )

    EXP_DIR.mkdir(exist_ok=True)
    pd.DataFrame(har_rows).set_index("asset").round(4).to_csv(EXP_DIR / "har_coefficients.csv")
    pd.DataFrame(garch_rows).set_index("asset").round(4).to_csv(EXP_DIR / "garch_coefficients.csv")
    with open(EXP_DIR / "baseline_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Console summary.
    print("\n=== HAR-RV coefficients ===")
    print(pd.DataFrame(har_rows).set_index("asset").round(3).to_string())
    print("\n=== GARCH(1,1) coefficients ===")
    print(pd.DataFrame(garch_rows).set_index("asset").round(4).to_string())
    print("\n=== Validation / Test log-RMSE and QLIKE ===")
    rows = []
    for t, m in all_metrics.items():
        rows.append({
            "asset": t,
            "HAR_val_rmse": m["HAR-RV/val/log_rmse"], "HAR_test_rmse": m["HAR-RV/test/log_rmse"],
            "HAR_val_qlike": m["HAR-RV/val/qlike"], "HAR_test_qlike": m["HAR-RV/test/qlike"],
            "GARCH_val_rmse": m["GARCH/val/log_rmse"], "GARCH_test_rmse": m["GARCH/test/log_rmse"],
        })
    print(pd.DataFrame(rows).set_index("asset").round(3).to_string())


if __name__ == "__main__":
    main()

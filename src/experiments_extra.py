"""Evaluation-driven robustness and ablation for the TCNN-LSTM (beyond the MEDIUM CUT).

The single-seed result is Diebold-Mariano significant, so two cheap checks are added to
support the claim responsibly: (1) a multi-seed robustness sweep to confirm the result is not
an artefact of one initialisation, and (2) a short-only / long-only branch ablation to test
whether the dual-resolution structure is actually doing work. Both log to the MLflow "neural"
experiment and write CSVs under experiments/. Run after train.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CONFIG, train_once

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
HAR_TEST_RMSE = 0.835
HAR_TEST_QLIKE = 0.445


def _canonical():
    m = json.load(open(EXP / "final_results.json"))["metrics"]["TCNN-LSTM"]
    return m["test_log_rmse"], m["test_qlike"]


def robustness(seeds=(1337, 2024, 7, 123)) -> pd.DataFrame:
    c_rmse, c_qlike = _canonical()
    rows = [{"seed": 42, "test_rmse": c_rmse, "test_qlike": c_qlike}]
    for s in seeds:
        r = train_once(dict(CONFIG, seed=s), run_name=f"robust_seed{s}",
                       log_mlflow=True, save_artifacts=False, verbose=False)
        rows.append({"seed": s, "test_rmse": r["test_rmse"], "test_qlike": r["test_qlike"]})
        print(f"  seed {s:4d}: test_rmse={r['test_rmse']:.3f}  test_qlike={r['test_qlike']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(EXP / "robustness.csv", index=False)
    return df


def ablation() -> pd.DataFrame:
    c_rmse, c_qlike = _canonical()
    rows = [{"branches": "dual", "test_rmse": c_rmse, "test_qlike": c_qlike}]
    for br in ("short", "long"):
        r = train_once(dict(CONFIG, branches=br), run_name=f"ablation_{br}_seed42",
                       log_mlflow=True, save_artifacts=False, verbose=False)
        rows.append({"branches": br, "test_rmse": r["test_rmse"], "test_qlike": r["test_qlike"]})
        print(f"  {br:5s}-only: test_rmse={r['test_rmse']:.3f}  test_qlike={r['test_qlike']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(EXP / "ablation.csv", index=False)
    return df


def main():
    print("=== Robustness: extra seeds (dual model) ===")
    rb = robustness()
    n = len(rb)
    print(f"\ntest log-RMSE over {n} seeds: mean={rb['test_rmse'].mean():.3f} "
          f"std={rb['test_rmse'].std():.3f} "
          f"[min {rb['test_rmse'].min():.3f}, max {rb['test_rmse'].max():.3f}]")
    print(f"seeds beating HAR-RV (RMSE {HAR_TEST_RMSE}): "
          f"{int((rb['test_rmse'] < HAR_TEST_RMSE).sum())}/{n}")
    print(f"seeds beating HAR-RV (QLIKE {HAR_TEST_QLIKE}): "
          f"{int((rb['test_qlike'] < HAR_TEST_QLIKE).sum())}/{n}")

    print("\n=== Ablation: branch contribution (seed 42) ===")
    ab = ablation()
    print(ab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

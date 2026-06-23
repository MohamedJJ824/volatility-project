"""Train the dual-resolution TCNN-LSTM on SPX log-RV (MEDIUM CUT: one asset, one seed).

Builds short (5-day) and long (22-day) input windows aligned to the same next-day target
and information set as HAR-RV, standardises with training-set statistics only, trains with
Adam + cosine schedule and early stopping on validation RMSE, logs to the MLflow "neural"
experiment, saves the best checkpoint, and writes test/val predictions for Phase 4.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E
from models import DualResTCNNLSTM

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "experiments" / "checkpoints"
EXP_DIR = ROOT / "experiments"
TRACKING_URI = "sqlite:///experiments/mlflow.db"

CONFIG = dict(
    asset="SPX",
    short_window=5,
    long_window=22,
    tcnn_channels=(32, 64),
    lstm_hidden=64,
    dropout=0.1,
    lr=1e-3,
    batch_size=64,
    max_epochs=200,
    patience=15,
    seed=42,
    branches="both",
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_windows(df: pd.DataFrame, short_w: int, long_w: int):
    """Return short/long windows, target, split label, and date for each predictable day."""
    x = df["log_rv"].to_numpy(dtype=np.float32)
    sp = df["split"].to_numpy()
    dt = df["date"].to_numpy()
    S, L, Y, SP, DT = [], [], [], [], []
    for i in range(long_w, len(x)):
        S.append(x[i - short_w:i])
        L.append(x[i - long_w:i])
        Y.append(x[i])
        SP.append(sp[i])
        DT.append(dt[i])
    return (np.asarray(S, np.float32), np.asarray(L, np.float32),
            np.asarray(Y, np.float32), np.asarray(SP), np.asarray(DT))


def to_tensor(arr):
    return torch.from_numpy(arr).unsqueeze(1)  # (N, 1, T)


def evaluate_split(model, Sz, Lz, y_log, mu, sigma, device):
    """Return (log-RMSE, QLIKE, pred_log) on a split, inverting standardisation."""
    model.eval()
    with torch.no_grad():
        pred_z = model(Sz.to(device), Lz.to(device)).cpu().numpy()
    pred_log = pred_z * sigma + mu
    rmse = E.rmse(y_log, pred_log)
    qlike = E.qlike(np.exp(y_log), np.exp(pred_log))
    return rmse, qlike, pred_log


def train_once(cfg: dict, run_name: str | None = None, log_mlflow: bool = True,
               save_artifacts: bool = False, verbose: bool = True) -> dict:
    """Train one TCNN-LSTM under cfg and return metrics, history, and predictions.

    save_artifacts writes the checkpoint, history, and prediction CSVs (canonical run only).
    """
    import mlflow

    set_seed(cfg["seed"])
    device = torch.device("cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROC / f"{cfg['asset']}.parquet").sort_values("date").reset_index(drop=True)
    S, L, Y, SP, DT = build_windows(df, cfg["short_window"], cfg["long_window"])

    # Standardise using TRAIN log-RV statistics only.
    mu = float(Y[SP == "train"].mean())
    sigma = float(Y[SP == "train"].std())
    Sz, Lz = (S - mu) / sigma, (L - mu) / sigma
    Yz = (Y - mu) / sigma

    idx = {sp: np.where(SP == sp)[0] for sp in ("train", "val", "test")}
    tr = idx["train"]
    Sz_t, Lz_t = to_tensor(Sz), to_tensor(Lz)
    Yz_t = torch.from_numpy(Yz)

    g = torch.Generator().manual_seed(cfg["seed"])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Sz_t[tr], Lz_t[tr], Yz_t[tr]),
        batch_size=cfg["batch_size"], shuffle=True, generator=g,
    )

    model = DualResTCNNLSTM(
        cfg["short_window"], cfg["long_window"], cfg["tcnn_channels"],
        cfg["lstm_hidden"], cfg["dropout"], branches=cfg.get("branches", "both"),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
    loss_fn = nn.MSELoss()

    best_val, best_state, best_epoch, wait = np.inf, None, -1, 0
    history = []
    if verbose:
        print(f"[train] {run_name or cfg['seed']}  branches={cfg.get('branches','both')}  "
              f"params={model.count_params()}")

    if log_mlflow:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("neural")
        run_ctx = mlflow.start_run(run_name=run_name)
    else:
        run_ctx = _nullcontext()
    with run_ctx:
        if log_mlflow:
            mlflow.log_params({k: (str(v) if isinstance(v, tuple) else v) for k, v in cfg.items()})
            mlflow.log_param("n_params", model.count_params())

        for epoch in range(cfg["max_epochs"]):
            model.train()
            ep_loss = 0.0
            for sb, lb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(sb.to(device), lb.to(device)), yb.to(device))
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(yb)
            ep_loss /= len(tr)
            sched.step()

            val_rmse, _, _ = evaluate_split(
                model, Sz_t[idx["val"]], Lz_t[idx["val"]], Y[idx["val"]], mu, sigma, device)
            history.append({"epoch": epoch, "train_loss": ep_loss, "val_rmse": val_rmse})
            if log_mlflow:
                mlflow.log_metrics({"train_loss": ep_loss, "val_rmse": val_rmse}, step=epoch)

            if val_rmse < best_val - 1e-5:
                best_val, best_epoch, wait = val_rmse, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= cfg["patience"]:
                break

        model.load_state_dict(best_state)
        val_rmse, val_qlike, val_pred = evaluate_split(
            model, Sz_t[idx["val"]], Lz_t[idx["val"]], Y[idx["val"]], mu, sigma, device)
        test_rmse, test_qlike, test_pred = evaluate_split(
            model, Sz_t[idx["test"]], Lz_t[idx["test"]], Y[idx["test"]], mu, sigma, device)

        if log_mlflow:
            mlflow.log_metrics({
                "best_epoch": best_epoch,
                "val_log_rmse": val_rmse, "val_qlike": val_qlike,
                "test_log_rmse": test_rmse, "test_qlike": test_qlike,
            })

        if save_artifacts:
            ckpt = CKPT_DIR / f"tcnnlstm_seed{cfg['seed']}.pt"
            torch.save({"state_dict": best_state, "mu": mu, "sigma": sigma, "config": cfg}, ckpt)
            if log_mlflow:
                mlflow.log_artifact(str(ckpt))
            pd.DataFrame(history).to_csv(EXP_DIR / "neural_history.csv", index=False)
            preds = [pd.DataFrame({"date": DT[idx[sp]], "split": sp,
                                   "y_log": Y[idx[sp]], "tcnn_pred_log": pr})
                     for sp, pr in (("val", val_pred), ("test", test_pred))]
            pd.concat(preds).to_csv(EXP_DIR / "neural_predictions.csv", index=False)

    return {"best_epoch": best_epoch, "val_rmse": val_rmse, "val_qlike": val_qlike,
            "test_rmse": test_rmse, "test_qlike": test_qlike, "history": history}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    res = train_once(CONFIG, run_name=f"SPX_TCNN-LSTM_seed{CONFIG['seed']}",
                     log_mlflow=True, save_artifacts=True)
    print("\n=== TCNN-LSTM (SPX, seed 42) ===")
    print(f"best epoch    : {res['best_epoch']}")
    print(f"val  log-RMSE : {res['val_rmse']:.3f}   (HAR-RV 0.896, GARCH 0.947)")
    print(f"val  QLIKE    : {res['val_qlike']:.3f}   (HAR-RV 0.515, GARCH 0.457)")
    print(f"test log-RMSE : {res['test_rmse']:.3f}   (HAR-RV 0.835, GARCH 0.884)")
    print(f"test QLIKE    : {res['test_qlike']:.3f}   (HAR-RV 0.445, GARCH 0.454)")


if __name__ == "__main__":
    main()

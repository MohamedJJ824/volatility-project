"""Data ingestion and realized-variance computation for the IE412 volatility project.

Two ingestion paths are provided:

1. load_oxford_man(ticker): the Oxford-Man Institute Realized Library, which ships
   pre-computed 5-minute realized variance. This is the primary plan in PROJECT_PLAN.md.
2. load_yfinance_gk(ticker): daily OHLC from yfinance with a Garman-Klass range-based
   variance estimator as an RV proxy. This is the fallback if Oxford-Man is unavailable.

Both functions return a DataFrame with columns [date, rv, log_rv] (plus a 'ret' column
of daily log returns where available, used by the GARCH baseline in Phase 2).

Running this module as a script downloads every ticker, applies the train/val/test
split, and writes data/processed/<TICKER>.parquet.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Configuration ----------------------------------------------------------

# Project tickers and their symbol in each data source.
#   key            -> short name used for output files
#   oxford         -> Oxford-Man Realized Library symbol
#   yahoo          -> yfinance / Yahoo Finance symbol
TICKERS = {
    "SPX": {"oxford": ".SPX", "yahoo": "^GSPC"},
    "FTSE": {"oxford": ".FTSE", "yahoo": "^FTSE"},
    "N225": {"oxford": ".N225", "yahoo": "^N225"},
    "DAX": {"oxford": ".GDAXI", "yahoo": "^GDAXI"},
}

# Split boundaries from PROJECT_PLAN.md Phase 1.
# COVID (2020) sits in validation by design as a stress test.
SPLITS = {
    "train": ("2000-01-01", "2018-12-31"),
    "val": ("2019-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2025-12-31"),
}

# Oxford-Man Realized Library CSV. The Institute discontinued public hosting in 2022,
# so this URL may 404; load_oxford_man surfaces a clear error and the caller falls back.
OXFORD_MAN_URL = (
    "https://raw.githubusercontent.com/onnokleen/mfGARCH/master/data-raw/"
    "oxfordmanrealizedvolatilityindices.csv"
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# RV values below this floor are treated as numerical noise before taking logs.
RV_FLOOR = 1e-10


# --- Primary path: Oxford-Man Realized Library ------------------------------

def load_oxford_man(ticker: str, rv_column: str = "rv5") -> pd.DataFrame:
    """Load pre-computed realized variance for one ticker from the Oxford-Man library.

    Parameters
    ----------
    ticker : str
        Project ticker key (e.g. "SPX"); mapped to the Oxford-Man symbol internally.
    rv_column : str
        Which realized-measure column to use. "rv5" is 5-minute realized variance.

    Returns
    -------
    DataFrame with columns [date, rv, log_rv, ret].

    Raises
    ------
    RuntimeError if the library cannot be downloaded or the schema is unexpected.
    """
    symbol = TICKERS[ticker]["oxford"]
    try:
        raw = pd.read_csv(OXFORD_MAN_URL, low_memory=False)
    except Exception as exc:  # network error, 404, parse failure
        raise RuntimeError(
            f"Oxford-Man library unavailable ({exc!r}). Use load_yfinance_gk fallback."
        ) from exc

    # The library is long-format: a 'Symbol' column plus per-date realized measures.
    sym_col = "Symbol" if "Symbol" in raw.columns else raw.columns[1]
    date_col = raw.columns[0]
    if rv_column not in raw.columns:
        raise RuntimeError(
            f"Oxford-Man schema changed: column {rv_column!r} not found. "
            f"Available: {list(raw.columns)[:10]}..."
        )

    df = raw.loc[raw[sym_col] == symbol, [date_col, rv_column]].copy()
    if df.empty:
        raise RuntimeError(f"No Oxford-Man rows for symbol {symbol!r}.")
    df.columns = ["date", "rv"]

    # open-to-close return if the library provides it (used by GARCH later).
    if "open_to_close" in raw.columns:
        ret = raw.loc[raw[sym_col] == symbol, "open_to_close"].to_numpy()
        df["ret"] = ret
    else:
        df["ret"] = np.nan

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df = _finalize(df)
    return df


# --- Fallback path: yfinance + Garman-Klass ---------------------------------

def garman_klass(o, h, l, c) -> np.ndarray:
    """Garman-Klass daily variance estimator (range-based RV proxy).

    sigma^2 = 0.5 * (ln(H/L))^2 - (2 ln2 - 1) * (ln(C/O))^2

    Returns an array of daily variance estimates in squared-return units.
    """
    o, h, l, c = (np.asarray(x, dtype=float) for x in (o, h, l, c))
    hl = np.log(h / l)
    co = np.log(c / o)
    return 0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2


def load_yfinance_gk(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    """Load daily OHLC from yfinance and build a Garman-Klass RV-proxy series.

    Returns
    -------
    DataFrame with columns [date, rv, log_rv, ret].
    """
    import yfinance as yf

    symbol = TICKERS[ticker]["yahoo"]
    raw = yf.download(
        symbol, start=start, auto_adjust=False, progress=False, multi_level_index=False
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol!r}.")

    raw = raw.rename(columns=str.lower)
    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise RuntimeError(f"yfinance missing columns {missing} for {symbol!r}.")

    gk = garman_klass(raw["open"], raw["high"], raw["low"], raw["close"])
    ret = np.log(raw["close"]).diff().to_numpy()

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.index).tz_localize(None).normalize(),
            "rv": gk,
            "ret": ret,
        }
    )
    df = _finalize(df)
    return df


# --- Shared cleaning --------------------------------------------------------

def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an [date, rv, ret] frame: drop bad RV, floor, add log_rv, sort, dedupe."""
    df = df.dropna(subset=["rv"]).copy()
    df = df[np.isfinite(df["rv"])]
    df = df[df["rv"] > 0]
    df["rv"] = df["rv"].clip(lower=RV_FLOOR)
    df["log_rv"] = np.log(df["rv"])
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df[["date", "rv", "log_rv", "ret"]]


def add_split_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with 'train' / 'val' / 'test' / NaN per SPLITS boundaries."""
    df = df.copy()
    df["split"] = pd.Series([None] * len(df), index=df.index, dtype="object")
    for name, (lo, hi) in SPLITS.items():
        mask = (df["date"] >= pd.Timestamp(lo)) & (df["date"] <= pd.Timestamp(hi))
        df.loc[mask, "split"] = name
    return df


def summary_stats(df: pd.DataFrame) -> dict:
    """Basic distributional + persistence diagnostics on log-RV (for the report table)."""
    x = df["log_rv"].to_numpy()
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else np.nan
    return {
        "n": int(len(x)),
        "start": df["date"].iloc[0].date().isoformat(),
        "end": df["date"].iloc[-1].date().isoformat(),
        "mean_log_rv": float(np.mean(x)),
        "std_log_rv": float(np.std(x)),
        "skew_log_rv": float(pd.Series(x).skew()),
        "kurt_log_rv": float(pd.Series(x).kurt()),
        "lag1_autocorr": float(lag1),
    }


def load_ticker(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Dispatch to a data source. source in {"oxford", "yfinance"}."""
    if source == "oxford":
        return load_oxford_man(ticker)
    if source == "yfinance":
        return load_yfinance_gk(ticker)
    raise ValueError(f"Unknown source {source!r}.")


def build_all(source: str = "yfinance") -> dict:
    """Download every ticker, split, persist parquet, and return summary stats."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for ticker in TICKERS:
        print(f"[data] loading {ticker} via {source} ...", flush=True)
        df = load_ticker(ticker, source=source)
        df = add_split_labels(df)
        out = PROCESSED_DIR / f"{ticker}.parquet"
        df.to_parquet(out, index=False)
        stats[ticker] = summary_stats(df)
        n_by_split = df["split"].value_counts(dropna=True).to_dict()
        print(f"[data]   {ticker}: {len(df)} rows -> {out.name}  splits={n_by_split}")
    return stats


if __name__ == "__main__":
    # Default to yfinance fallback; pass "oxford" to try the primary library.
    src = sys.argv[1] if len(sys.argv) > 1 else "yfinance"
    s = build_all(source=src)
    print("\n=== Summary (log-RV) ===")
    for t, v in s.items():
        print(
            f"{t:5s} n={v['n']:5d}  mean={v['mean_log_rv']:+.3f}  "
            f"std={v['std_log_rv']:.3f}  lag1AC={v['lag1_autocorr']:.3f}  "
            f"[{v['start']}..{v['end']}]"
        )

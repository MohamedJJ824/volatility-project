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

"""Symbol-spaced feed-forward equalizer (FFE) with LMS adaptation."""
from __future__ import annotations

import numpy as np


def tap_matrix(s: np.ndarray, n_taps: int, n_pre: int) -> np.ndarray:
    """Rows are FFE input vectors: row k = [s[k+n_pre], ..., s[k+n_pre-n_taps+1]]."""
    return np.stack([np.roll(s, -(n_pre - i)) for i in range(n_taps)], axis=1)


def apply_ffe(s: np.ndarray, taps: np.ndarray, n_pre: int) -> np.ndarray:
    """y[k] = sum_i taps[i] * s[k + n_pre - i] (circular)."""
    return tap_matrix(s, len(taps), n_pre) @ taps


def lms_train(s: np.ndarray, desired: np.ndarray, n_taps: int, n_pre: int,
              mu: float, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """Adapt FFE taps with normalized LMS on known training symbols.

    Returns the final taps and the squared error at each training step.
    """
    if not 0 <= n_pre < n_taps:
        raise ValueError("n_pre must be in [0, n_taps)")
    x = tap_matrix(s, n_taps, n_pre)
    taps = np.zeros(n_taps)
    taps[n_pre] = 1.0
    sq_err = np.empty(n_train)
    for k in range(n_train):
        xk = x[k]
        e = desired[k] - xk @ taps
        taps += mu * e * xk / (xk @ xk + 1e-12)
        sq_err[k] = e * e
    return taps, sq_err


def ls_gain(s: np.ndarray, desired: np.ndarray) -> float:
    """Least-squares scalar gain fitting ``s`` to ``desired`` (used when FFE is off)."""
    return float(desired @ s / (s @ s))

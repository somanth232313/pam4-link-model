"""Frequency-domain models of the link's linear blocks.

All responses are evaluated on an ``np.fft.rfftfreq`` axis and applied with
:func:`apply_response` (circular convolution, so the PRBS pattern is treated
as periodic and no edge transients appear).
"""
from __future__ import annotations

import numpy as np


def apply_response(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Filter real signal ``x`` with frequency response ``h`` (on rfft bins)."""
    return np.fft.irfft(np.fft.rfft(x) * h, n=len(x))


def butterworth(f: np.ndarray, fc: float | None, order: int = 2) -> np.ndarray:
    """Causal analog Butterworth low-pass (driver / TIA bandwidth). ``None`` = ideal."""
    if fc is None:
        return np.ones_like(f, dtype=complex)
    # left-half-plane poles on a circle of radius wc: H(s) = prod(-p_k / (s - p_k))
    wc = 2 * np.pi * fc
    poles = wc * np.exp(1j * np.pi * (2 * np.arange(order) + order + 1) / (2 * order))
    s = 2j * np.pi * np.asarray(f, dtype=float)
    h = np.ones_like(s)
    for p in poles:
        h = h * (-p / (s - p))
    return h


def ctle(f: np.ndarray, peaking_db: float, f_pole1: float, f_pole2: float) -> np.ndarray:
    """One-zero, two-pole CTLE with unity DC gain.

    The zero sits ``peaking_db`` below the first pole, so the high-frequency
    boost relative to DC is ``peaking_db``. ``peaking_db <= 0`` bypasses it.
    """
    if peaking_db <= 0:
        return np.ones_like(f, dtype=complex)
    f_zero = f_pole1 / 10 ** (peaking_db / 20)
    s = 1j * f
    return (1 + s / f_zero) / ((1 + s / f_pole1) * (1 + s / f_pole2))


def channel_loss(f: np.ndarray, loss_db_at_nyquist: float, f_nyquist: float,
                 skin_fraction: float = 0.5) -> np.ndarray:
    """Electrical channel insertion loss (package + PCB trace).

    Loss in dB grows as a mix of sqrt(f) (skin effect) and f (dielectric),
    scaled to ``loss_db_at_nyquist``. Zero-phase magnitude model.
    """
    if loss_db_at_nyquist <= 0:
        return np.ones_like(f, dtype=complex)
    x = f / f_nyquist
    loss_db = loss_db_at_nyquist * (skin_fraction * np.sqrt(x) + (1 - skin_fraction) * x)
    return (10 ** (-loss_db / 20)).astype(complex)

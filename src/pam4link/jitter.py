"""Sampling-clock jitter and jittered sampling."""
from __future__ import annotations

import numpy as np


def jitter_sequence(n: int, baud: float, rj_rms_ui: float, sj_pp_ui: float,
                    sj_freq_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Per-symbol sampling-time error in UI.

    Random jitter (RJ) is Gaussian; deterministic jitter is modeled as
    sinusoidal jitter (SJ) with peak-to-peak ``sj_pp_ui`` at ``sj_freq_hz``.
    """
    j = rj_rms_ui * rng.standard_normal(n)
    if sj_pp_ui > 0:
        k = np.arange(n)
        j += 0.5 * sj_pp_ui * np.sin(2 * np.pi * sj_freq_hz * k / baud + rng.uniform(0, 2 * np.pi))
    return j


def sample_at(wave: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Linearly interpolate a periodic waveform at fractional sample indices."""
    return np.interp(times, np.arange(len(wave)), wave, period=len(wave))

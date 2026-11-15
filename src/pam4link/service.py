"""JSON-in / JSON-out wrapper around the simulator for the web demo.

NumPy only (no Matplotlib) so it fits comfortably in a serverless function.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .jitter import sample_at
from .link import KP4_BER_THRESHOLD, LinkConfig, LinkResult, bathtub, eye_width, simulate

# name -> (default, min, max)
PARAMS: dict[str, tuple[float, float, float]] = {
    "rx_power_dbm": (-6.0, -14.0, -2.0),
    "host_channel_loss_db": (10.0, 0.0, 16.0),
    "ctle_peaking_db": (6.0, 0.0, 12.0),
    "ffe_taps": (7, 0, 11),
    "tia_noise_pa": (15.0, 5.0, 30.0),
    "extinction_ratio_db": (5.0, 3.0, 8.0),
    "rj_rms_ui": (0.01, 0.0, 0.05),
    "sj_pp_ui": (0.05, 0.0, 0.3),
}

WEB_BASE = LinkConfig(sps=16, n_symbols=2**14, ffe_train=4000)
SWEEP_POWERS = np.arange(-14.0, -1.9, 1.0)
EYE_ROWS = 96


def parse_params(raw: dict[str, Any] | None) -> dict[str, float]:
    """Fill defaults and clamp every value to its allowed range."""
    raw = raw or {}
    out: dict[str, float] = {}
    for name, (default, lo, hi) in PARAMS.items():
        try:
            v = float(raw.get(name, default))
        except (TypeError, ValueError):
            v = float(default)
        if not np.isfinite(v):
            v = float(default)
        out[name] = min(max(v, lo), hi)
    taps = int(round(out["ffe_taps"]))
    out["ffe_taps"] = 0 if taps < 2 else taps  # 0 = bypass; otherwise >= 2 taps
    return out


def config_from_params(p: dict[str, float], base: LinkConfig = WEB_BASE) -> LinkConfig:
    taps = int(p["ffe_taps"])
    return replace(
        base,
        fiber_loss_db=base.launch_power_dbm - p["rx_power_dbm"],
        host_channel_loss_db=p["host_channel_loss_db"],
        ctle_peaking_db=p["ctle_peaking_db"],
        ffe_taps=taps,
        ffe_pre=min(2, max(taps - 1, 0) // 2) if taps else 0,
        tia_noise_density=p["tia_noise_pa"] * 1e-12,
        extinction_ratio_db=p["extinction_ratio_db"],
        rj_rms_ui=p["rj_rms_ui"],
        sj_pp_ui=p["sj_pp_ui"],
    )


def _equalized_wave(res: LinkResult) -> np.ndarray:
    wave = res.clean_wave + res.noise_wave
    out = np.zeros_like(wave)
    for i, w in enumerate(res.ffe_taps):
        out += w * np.roll(wave, -(res.ffe_pre - i) * res.cfg.sps)
    return out


def eye_histogram(res: LinkResult, wave: np.ndarray, n_traces: int = 3000) -> dict[str, Any]:
    """2-UI eye density (rows = amplitude bins, top row = highest amplitude)."""
    cfg = res.cfg
    sps = cfg.sps
    start = min(cfg.ffe_train, cfg.n_symbols // 2)
    ks = np.arange(start, min(cfg.n_symbols - 4, start + n_traces))
    tt = np.arange(-sps, sps)
    times = ks[:, None] * sps + res.sample_offset + tt[None, :] + res.jitter_ui[ks, None] * sps
    seg = sample_at(wave, times.ravel()).reshape(times.shape)
    lim = float(np.percentile(np.abs(seg), 99.9) * 1.1)
    hist, _, _ = np.histogram2d(
        np.broadcast_to(np.arange(2 * sps), seg.shape).ravel(), seg.ravel(),
        bins=[2 * sps, EYE_ROWS], range=[[-0.5, 2 * sps - 0.5], [-lim, lim]],
    )
    return {"cols": 2 * sps, "rows": EYE_ROWS, "y_max": lim,
            "density": np.flipud(hist.T).astype(int).ravel().tolist()}


def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def run(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Simulate the link for the given UI parameters and return everything the page draws."""
    p = parse_params(raw)
    cfg = config_from_params(p)
    res = simulate(cfg)

    phases = np.linspace(-0.5, 0.5, 41)
    tub = bathtub(res, phases)

    sweep = []
    for pw in SWEEP_POWERS:
        r = simulate(replace(cfg, n_symbols=2**12, ffe_train=1500, fiber_loss_db=cfg.launch_power_dbm - pw))
        sweep.append(r.ber_statistical)

    ex = res.extras
    f = ex["frequency_hz"]
    idx = np.linspace(0, np.searchsorted(f, 2 * cfg.baud) - 1, 120).astype(int)
    db = lambda h: (20 * np.log10(np.abs(h[idx]) + 1e-15)).round(2).tolist()  # noqa: E731

    rx_wave = res.clean_wave + res.noise_wave
    return {
        "params": p,
        "metrics": {
            "ber_monte_carlo": res.ber,
            "bit_errors": res.bit_errors,
            "bits": res.n_bits,
            "ber_statistical": res.ber_statistical,
            "eye_opening": res.eye_opening(after_ffe=True),
            "eye_opening_before_ffe": res.eye_opening(after_ffe=False),
            "eye_width_ui": eye_width(phases, tub),
            "noise_sigma": res.sigma,
            "rx_power_dbm": cfg.rx_power_dbm,
            "kp4_threshold": KP4_BER_THRESHOLD,
            "passes_kp4": bool(res.ber_statistical < KP4_BER_THRESHOLD),
        },
        "eye_rx": eye_histogram(res, rx_wave),
        "eye_eq": eye_histogram(res, _equalized_wave(res)) if cfg.ffe_taps else None,
        "bathtub": {"phase_ui": phases.round(4).tolist(), "ber": [_finite(b) for b in tub]},
        "ber_vs_power": {"rx_power_dbm": SWEEP_POWERS.tolist(), "ber": [_finite(b) for b in sweep]},
        "ffe": {"taps": res.ffe_taps.round(4).tolist(), "pre": res.ffe_pre},
        "frequency_response": {
            "f_ghz": (f[idx] / 1e9).round(3).tolist(),
            "tx_db": db(ex["h_tx"]), "rx_db": db(ex["h_rx"]), "total_db": db(ex["h_tx"] * ex["h_rx"]),
        },
    }

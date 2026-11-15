"""Matplotlib figures: eye diagrams, bathtubs, BER curves, responses, FFE."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .jitter import sample_at
from .link import KP4_BER_THRESHOLD, LinkResult


def equalized_wave(res: LinkResult) -> np.ndarray:
    """Continuous-time view of the FFE output (taps spaced one UI apart)."""
    wave = res.clean_wave + res.noise_wave
    sps = res.cfg.sps
    out = np.zeros_like(wave)
    for i, w in enumerate(res.ffe_taps):
        out += w * np.roll(wave, -(res.ffe_pre - i) * sps)
    return out


def plot_eye(ax: Axes, res: LinkResult, wave: np.ndarray | None = None, title: str = "",
             n_traces: int = 4000) -> None:
    """2-UI eye diagram as a density plot, re-timed by the sampling-clock jitter."""
    cfg = res.cfg
    if wave is None:
        wave = res.clean_wave + res.noise_wave
    sps = cfg.sps
    ks = np.arange(cfg.ffe_train, min(cfg.n_symbols - 4, cfg.ffe_train + n_traces))
    tt = np.arange(-sps, sps)
    times = ks[:, None] * sps + res.sample_offset + tt[None, :] + res.jitter_ui[ks, None] * sps
    seg = sample_at(wave, times.ravel()).reshape(times.shape)
    xs = np.broadcast_to(tt / sps, seg.shape).ravel()
    lim = np.percentile(np.abs(seg), 99.9) * 1.1
    hist, xe, ye = np.histogram2d(xs, seg.ravel(), bins=[2 * sps, 220], range=[[-1, 1], [-lim, lim]])
    ax.imshow(np.log1p(hist.T), origin="lower", aspect="auto", cmap="magma",
              extent=[xe[0], xe[-1], ye[0], ye[-1]])
    ax.set_xlabel("time (UI, 0 = sampling point)")
    ax.set_ylabel("amplitude (normalized)")
    ax.set_title(title)


def plot_bathtub(ax: Axes, phases_ui: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    for label, ber in curves.items():
        ax.semilogy(phases_ui, np.clip(ber, 1e-15, 1), label=label)
    ax.axhline(KP4_BER_THRESHOLD, color="gray", ls="--", lw=1, label="KP4 FEC limit")
    ax.set_ylim(1e-12, 1)
    ax.set_xlabel("sampling phase offset (UI)")
    ax.set_ylabel("BER")
    ax.set_title("Bathtub curves (statistical)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def plot_ber_vs_power(ax: Axes, sweeps: dict[str, dict[str, np.ndarray]]) -> None:
    for (label, sw), color in zip(sweeps.items(), ["C0", "C1", "C2", "C3"]):
        ax.semilogy(sw["rx_power_dbm"], np.clip(sw["ber_stat"], 1e-15, 1), color=color, label=f"{label} (statistical)")
        mc = sw["ber_mc"]
        has = mc > 0
        ax.semilogy(sw["rx_power_dbm"][has], mc[has], "o", color=color, mfc="none", label=f"{label} (Monte Carlo)")
    ax.axhline(KP4_BER_THRESHOLD, color="gray", ls="--", lw=1, label="KP4 FEC limit")
    ax.set_ylim(1e-12, 1)
    ax.set_xlabel("received average optical power (dBm)")
    ax.set_ylabel("BER")
    ax.set_title("BER vs. received power")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def plot_responses(ax: Axes, res: LinkResult) -> None:
    ex = res.extras
    f = ex["frequency_hz"]
    keep = f <= 2 * res.cfg.baud
    db = lambda h: 20 * np.log10(np.abs(h[keep]) + 1e-15)  # noqa: E731
    ax.plot(f[keep] / 1e9, db(ex["h_tx"]), label="host channel + driver")
    ax.plot(f[keep] / 1e9, db(ex["h_rx"]), label="TIA + CTLE")
    ax.plot(f[keep] / 1e9, db(ex["h_tx"] * ex["h_rx"]), label="end to end", lw=2)
    ax.axvline(res.cfg.f_nyquist / 1e9, color="gray", ls="--", lw=1, label="Nyquist")
    ax.set_ylim(-40, 10)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title("Frequency responses")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_ffe(ax_taps: Axes, ax_conv: Axes, res: LinkResult, window: int = 200) -> None:
    idx = np.arange(len(res.ffe_taps)) - res.ffe_pre
    ax_taps.stem(idx, res.ffe_taps)
    ax_taps.set_xlabel("tap (UI relative to main cursor)")
    ax_taps.set_title("Adapted FFE taps")
    ax_taps.grid(True, alpha=0.3)
    if res.lms_sq_error is not None:
        mse = np.convolve(res.lms_sq_error, np.ones(window) / window, mode="valid")
        ax_conv.plot(10 * np.log10(mse))
        ax_conv.set_xlabel("training symbol")
        ax_conv.set_ylabel("MSE (dB)")
        ax_conv.set_title("LMS convergence")
        ax_conv.grid(True, alpha=0.3)


def slicer_samples(res: LinkResult) -> np.ndarray:
    """Noisy slicer-input samples (for histograms)."""
    return (res.y_clean + res.y_noise)[res.eval_slice]


__all__ = ["equalized_wave", "plot_eye", "plot_bathtub", "plot_ber_vs_power", "plot_responses",
           "plot_ffe", "slicer_samples"]

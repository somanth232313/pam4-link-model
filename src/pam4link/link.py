"""End-to-end 106.25 Gb/s PAM4 electro-optical link simulation.

Signal chain::

    PRBS -> PAM4 (Gray) -> DAC -> host channel loss -> modulator driver
    (bandwidth + compression) -> optical modulator (extinction ratio)
    -> fiber loss -> photodiode -> + TIA input noise -> TIA bandwidth
    -> CTLE -> jittered sampling -> LMS FFE -> slicer -> BER

Every block after the noise is linear, so the receiver waveform is tracked
as ``clean + noise``. That lets the same run report a Monte Carlo BER and a
statistical BER (noiseless samples + Gaussian noise sigma).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .equalizer import apply_ffe, lms_train, ls_gain
from .filters import apply_response, butterworth, channel_loss, ctle
from .jitter import jitter_sequence, sample_at
from .pam4 import LEVELS, bits_to_symbols, slice_symbols, statistical_ber, symbols_to_bits
from .prbs import prbs

KP4_BER_THRESHOLD = 2.4e-4  # pre-FEC BER limit of RS(544,514) "KP4" FEC


@dataclass(frozen=True)
class LinkConfig:
    # signaling
    baud: float = 53.125e9            # 106.25 Gb/s PAM4
    sps: int = 32                     # waveform samples per UI
    n_symbols: int = 2**16
    prbs_order: int = 31
    seed: int = 1
    # transmitter
    host_channel_loss_db: float = 10.0   # electrical loss at Nyquist before the driver
    driver_bw_hz: float | None = 32e9
    driver_compression: float = 0.5      # tanh compression strength, 0 = linear
    # optics
    launch_power_dbm: float = 0.0        # average optical power out of the modulator
    extinction_ratio_db: float = 5.0
    fiber_loss_db: float = 6.0
    # receiver
    responsivity_a_per_w: float = 0.8
    tia_bw_hz: float | None = 35e9
    tia_noise_density: float = 15e-12    # input-referred noise, A/sqrt(Hz)
    ctle_peaking_db: float = 6.0         # 0 = CTLE bypassed
    ctle_pole1_hz: float | None = None   # default: Nyquist
    ctle_pole2_hz: float | None = None   # default: baud rate
    # clock
    rj_rms_ui: float = 0.01
    sj_pp_ui: float = 0.05
    sj_freq_hz: float = 20e6
    # FFE
    ffe_taps: int = 7                    # 0 = FFE bypassed
    ffe_pre: int = 2                     # pre-cursor taps
    ffe_mu: float = 0.02
    ffe_train: int = 8000

    @property
    def fs(self) -> float:
        return self.baud * self.sps

    @property
    def f_nyquist(self) -> float:
        return self.baud / 2

    @property
    def rx_power_dbm(self) -> float:
        """Average optical power at the photodiode."""
        return self.launch_power_dbm - self.fiber_loss_db


@dataclass
class LinkResult:
    cfg: LinkConfig
    tx_symbols: np.ndarray       # symbol indices 0..3
    clean_wave: np.ndarray       # receiver waveform before sampling, noiseless
    noise_wave: np.ndarray       # receiver noise waveform (same scaling)
    sample_offset: float         # sampling point within the waveform, in samples
    jitter_ui: np.ndarray        # per-symbol sampling jitter
    ffe_taps: np.ndarray
    ffe_pre: int
    lms_sq_error: np.ndarray | None
    s_clean: np.ndarray          # sampled, before FFE
    s_noise: np.ndarray
    y_clean: np.ndarray          # slicer input, noiseless
    y_noise: np.ndarray          # slicer input, noise only
    eval_slice: slice            # symbols counted for BER (excludes training)
    bit_errors: int = 0
    n_bits: int = 0
    extras: dict = field(default_factory=dict)

    @property
    def ber(self) -> float:
        return self.bit_errors / self.n_bits

    @property
    def sigma(self) -> float:
        return float(np.std(self.y_noise[self.eval_slice]))

    @property
    def ber_statistical(self) -> float:
        e = self.eval_slice
        return statistical_ber(self.y_clean[e], self.tx_symbols[e], self.sigma)

    def eye_opening(self, after_ffe: bool = True) -> float:
        """Vertical eye opening of the noiseless samples (1.0 = ideal, <= 0 = closed)."""
        e = self.eval_slice
        if after_ffe:
            y = self.y_clean[e]
        else:
            y = self.s_clean[e] * ls_gain(self.s_clean[e], LEVELS[self.tx_symbols[e]])
        return vertical_eye_opening(y, self.tx_symbols[e])


def vertical_eye_opening(y: np.ndarray, symbols: np.ndarray) -> float:
    """Smallest of the three inner eye heights, normalized to the ideal height of 2."""
    heights = [y[symbols == i + 1].min() - y[symbols == i].max() for i in range(3)]
    return float(min(heights) / 2.0)


def _eye_metric(samples: np.ndarray, symbols: np.ndarray) -> float:
    """Worst-eye separation between adjacent level clusters, in units of their spread."""
    vals = []
    for i in range(3):
        lo, hi = samples[symbols == i], samples[symbols == i + 1]
        vals.append((hi.mean() - lo.mean()) / (hi.std() + lo.std() + 1e-12))
    return min(vals)


def find_sampling_offset(clean: np.ndarray, tx_wave: np.ndarray, symbols: np.ndarray, sps: int) -> float:
    """Ideal-CDR sampling point: coarse delay by cross-correlation, then the best phase in that UI."""
    n_wave = len(clean)
    corr = np.fft.irfft(np.fft.rfft(clean) * np.conj(np.fft.rfft(tx_wave)), n=n_wave)
    delay = int(np.argmax(corr))
    if delay > n_wave // 2:
        delay -= n_wave
    center = delay + (sps - 1) / 2
    candidates = center + np.arange(-sps / 2, sps / 2 + 0.25, 0.5)
    k = np.arange(len(symbols)) * sps
    scores = np.array([_eye_metric(sample_at(clean, k + c), symbols) for c in candidates])
    # among (near-)equal best scores, prefer the one closest to the UI center
    good = np.flatnonzero(scores >= scores.max() - 1e-9 * abs(scores.max()))
    return float(candidates[good[np.argmin(np.abs(candidates[good] - center))]])


def simulate(cfg: LinkConfig) -> LinkResult:
    """Run one link simulation and count bit errors."""
    rng = np.random.default_rng(cfg.seed)
    n, sps = cfg.n_symbols, cfg.sps

    # --- transmitter -------------------------------------------------------
    bits = prbs(cfg.prbs_order, 2 * n, seed=int(rng.integers(1, 2**cfg.prbs_order)))
    symbols = bits_to_symbols(bits)
    tx_wave = np.repeat(LEVELS[symbols], sps)
    f = np.fft.rfftfreq(len(tx_wave), d=1 / cfg.fs)

    h_tx = channel_loss(f, cfg.host_channel_loss_db, cfg.f_nyquist) * butterworth(f, cfg.driver_bw_hz)
    drive = apply_response(tx_wave, h_tx) / 3.0          # outer levels at +/-1
    if cfg.driver_compression > 0:
        k = cfg.driver_compression
        drive = np.tanh(k * drive) / np.tanh(k)

    # --- optics ------------------------------------------------------------
    er = 10 ** (cfg.extinction_ratio_db / 10)
    mod_index = (er - 1) / (er + 1)
    p_avg = 1e-3 * 10 ** (cfg.launch_power_dbm / 10)
    p_tx = np.clip(p_avg * (1 + mod_index * drive), 0.0, None)
    p_rx = p_tx * 10 ** (-cfg.fiber_loss_db / 10)

    # --- receiver front end -----------------------------------------------
    i_pd = cfg.responsivity_a_per_w * p_rx
    i_noise = cfg.tia_noise_density * np.sqrt(cfg.fs / 2) * rng.standard_normal(len(i_pd))
    h_rx = butterworth(f, cfg.tia_bw_hz) * ctle(
        f, cfg.ctle_peaking_db,
        cfg.ctle_pole1_hz or cfg.f_nyquist,
        cfg.ctle_pole2_hz or cfg.baud,
    )
    clean = apply_response(i_pd - i_pd.mean(), h_rx)
    noise = apply_response(i_noise, h_rx)
    agc = np.sqrt(np.mean(LEVELS**2)) / clean.std()      # AGC: same gain for signal and noise
    clean *= agc
    noise *= agc

    # --- clock recovery + sampling ----------------------------------------
    offset = find_sampling_offset(clean, tx_wave, symbols, sps)
    jit = jitter_sequence(n, cfg.baud, cfg.rj_rms_ui, cfg.sj_pp_ui, cfg.sj_freq_hz, rng)
    t = np.arange(n) * sps + offset + jit * sps
    s_clean, s_noise = sample_at(clean, t), sample_at(noise, t)

    # --- equalizer ----------------------------------------------------------
    desired = LEVELS[symbols]
    n_train = min(cfg.ffe_train, n // 2)
    if cfg.ffe_taps > 0:
        taps, sq_err = lms_train(s_clean + s_noise, desired, cfg.ffe_taps, cfg.ffe_pre, cfg.ffe_mu, n_train)
        pre = cfg.ffe_pre
    else:
        taps, sq_err, pre = np.array([ls_gain((s_clean + s_noise)[:n_train], desired[:n_train])]), None, 0
    y_clean, y_noise = apply_ffe(s_clean, taps, pre), apply_ffe(s_noise, taps, pre)

    # --- slicer + BER (skip training symbols and circular-wrap edge) -------
    ev = slice(n_train, n - len(taps))
    rx_symbols = slice_symbols(y_clean + y_noise)
    tx_bits, rx_bits = symbols_to_bits(symbols[ev]), symbols_to_bits(rx_symbols[ev])

    return LinkResult(
        cfg=cfg, tx_symbols=symbols, clean_wave=clean, noise_wave=noise,
        sample_offset=offset, jitter_ui=jit, ffe_taps=taps, ffe_pre=pre, lms_sq_error=sq_err,
        s_clean=s_clean, s_noise=s_noise, y_clean=y_clean, y_noise=y_noise, eval_slice=ev,
        bit_errors=int(np.count_nonzero(tx_bits != rx_bits)), n_bits=len(tx_bits),
        extras={"frequency_hz": f, "h_tx": h_tx, "h_rx": h_rx},
    )


def bathtub(res: LinkResult, phases_ui: np.ndarray) -> np.ndarray:
    """Statistical BER vs. sampling-phase offset (UI), with the link's jitter and frozen FFE taps."""
    cfg, e = res.cfg, res.eval_slice
    k = np.arange(cfg.n_symbols) * cfg.sps
    out = np.empty(len(phases_ui))
    for i, ph in enumerate(phases_ui):
        s = sample_at(res.clean_wave, k + res.sample_offset + (ph + res.jitter_ui) * cfg.sps)
        y = apply_ffe(s, res.ffe_taps, res.ffe_pre)
        out[i] = statistical_ber(y[e], res.tx_symbols[e], res.sigma)
    return out


def eye_width(phases_ui: np.ndarray, ber: np.ndarray, target: float = KP4_BER_THRESHOLD) -> float:
    """Horizontal eye opening (UI) where the bathtub stays below ``target``."""
    ok = ber < target
    if not ok.any():
        return 0.0
    step = phases_ui[1] - phases_ui[0]
    return float(ok.sum() * step)


def sweep_rx_power(cfg: LinkConfig, fiber_losses_db) -> dict[str, np.ndarray]:
    """BER (Monte Carlo and statistical) vs. received average optical power."""
    rx, mc, stat = [], [], []
    for loss in fiber_losses_db:
        r = simulate(replace(cfg, fiber_loss_db=float(loss)))
        rx.append(r.cfg.rx_power_dbm)
        mc.append(r.ber)
        stat.append(r.ber_statistical)
    return {"rx_power_dbm": np.array(rx), "ber_mc": np.array(mc), "ber_stat": np.array(stat)}


def sensitivity_dbm(rx_power_dbm: np.ndarray, ber: np.ndarray, target: float = KP4_BER_THRESHOLD) -> float:
    """Received power where BER crosses ``target`` (log-linear interpolation); NaN if never crossed."""
    logb = np.log10(np.clip(ber, 1e-30, None))
    order = np.argsort(rx_power_dbm)
    p, lb = rx_power_dbm[order], logb[order]
    below = np.flatnonzero(lb < np.log10(target))
    if len(below) == 0 or below[0] == 0:
        return float("nan")
    i = below[0]
    return float(np.interp(np.log10(target), [lb[i], lb[i - 1]], [p[i], p[i - 1]]))

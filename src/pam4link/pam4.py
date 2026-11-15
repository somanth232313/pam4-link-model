"""PAM4 symbol mapping, slicing and Gaussian-noise BER theory."""
from __future__ import annotations

import numpy as np

LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])
THRESHOLDS = np.array([-2.0, 0.0, 2.0])

# Gray code: symbol index -> (msb, lsb). Adjacent levels differ by one bit.
GRAY_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
_PAIR_TO_SYMBOL = np.array([0, 1, 3, 2])  # indexed by 2*msb + lsb


def bits_to_symbols(bits: np.ndarray) -> np.ndarray:
    """Map bit pairs (msb, lsb) to Gray-coded PAM4 symbol indices 0..3."""
    bits = np.asarray(bits, dtype=np.uint8)
    pairs = bits[: len(bits) // 2 * 2].reshape(-1, 2)
    return _PAIR_TO_SYMBOL[2 * pairs[:, 0] + pairs[:, 1]]


def symbols_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Inverse of :func:`bits_to_symbols`."""
    return GRAY_BITS[np.asarray(symbols)].reshape(-1)


def slice_symbols(y: np.ndarray) -> np.ndarray:
    """Decide symbol indices from samples scaled to levels {-3, -1, 1, 3}."""
    return np.digitize(y, THRESHOLDS)


def erfc(x):
    """Complementary error function, relative error < 1.2e-7 everywhere (Numerical Recipes erfcc).

    Kept in NumPy so the package has no SciPy dependency (smaller serverless bundle).
    """
    x = np.asarray(x, dtype=float)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277))))))))
    with np.errstate(over="ignore", invalid="ignore"):
        r = np.where(np.isinf(z), 0.0, t * np.exp(poly))
    return np.where(x >= 0, r, 2.0 - r)


def q_func(x):
    """Gaussian tail probability Q(x)."""
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def q_inv(p):
    """Inverse of :func:`q_func` for 0 < p < 0.5 (Newton's method on ln Q)."""
    p = np.asarray(p, dtype=float)
    x = np.sqrt(-2.0 * np.log(p))  # asymptotic starting point, always above the root
    for _ in range(50):
        q = q_func(x)
        pdf = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
        x = x + (np.log(q) - np.log(p)) * q / pdf
    return x


def ber_from_q(q):
    """Gray-coded PAM4 BER for Gaussian noise, where q = (half level spacing) / sigma.

    SER = 2 (1 - 1/M) Q(q) = 1.5 Q(q); Gray coding gives one bit error per
    (adjacent) symbol error, over 2 bits per symbol, so BER = 0.75 Q(q).
    """
    return 0.75 * q_func(q)


def q_from_ber(ber):
    """Inverse of :func:`ber_from_q`."""
    return q_inv(np.asarray(ber) / 0.75)


def statistical_ber(y_clean: np.ndarray, symbols: np.ndarray, sigma: float) -> float:
    """Expected BER given noiseless slicer samples and Gaussian noise sigma.

    Each sample's error probability is the Gaussian tail beyond its nearest
    lower and upper decision thresholds, so ISI and jitter in ``y_clean`` are
    accounted for exactly; only the noise is treated statistically.
    """
    symbols = np.asarray(symbols)
    if sigma <= 0:  # noiseless: only samples already on the wrong side fail
        return float(np.mean(slice_symbols(y_clean) != symbols) / 2.0)
    lower = np.where(symbols > 0, THRESHOLDS[np.clip(symbols - 1, 0, 2)], -np.inf)
    upper = np.where(symbols < 3, THRESHOLDS[np.clip(symbols, 0, 2)], np.inf)
    p_sym = q_func((y_clean - lower) / sigma) + q_func((upper - y_clean) / sigma)
    return float(np.mean(p_sym) / 2.0)

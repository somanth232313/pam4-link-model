"""Pseudo-random binary sequence (PRBS) generators.

Fibonacci LFSRs using the ITU-T O.150 / IEEE 802.3 generator polynomials.
"""
from __future__ import annotations

import numpy as np

# order -> feedback taps of the generator polynomial x^a + x^b (+ ...) + 1
PRBS_TAPS: dict[int, tuple[int, ...]] = {
    7: (7, 6),
    9: (9, 5),
    11: (11, 9),
    13: (13, 12, 2, 1),  # PRBS13 used for PAM4 (IEEE 802.3bs PRBS13Q)
    15: (15, 14),
    23: (23, 18),
    31: (31, 28),
}


def prbs(order: int, n_bits: int, seed: int | None = None) -> np.ndarray:
    """Return ``n_bits`` bits of a PRBS of the given order as a uint8 array.

    ``seed`` sets the initial (non-zero) LFSR state; default is all ones.
    The sequence repeats every ``2**order - 1`` bits.
    """
    if order not in PRBS_TAPS:
        raise ValueError(f"unsupported PRBS order {order}; choose from {sorted(PRBS_TAPS)}")
    mask = (1 << order) - 1
    state = mask if seed is None else (seed & mask) or 1
    shifts = [t - 1 for t in PRBS_TAPS[order]]
    out = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        bit = 0
        for s in shifts:
            bit ^= (state >> s) & 1
        state = ((state << 1) | bit) & mask
        out[i] = bit
    return out

"""Unit tests for the individual link blocks."""
import numpy as np
import pytest

from pam4link.equalizer import apply_ffe, lms_train
from pam4link.filters import apply_response, butterworth, channel_loss, ctle
from pam4link.pam4 import (
    GRAY_BITS, LEVELS, ber_from_q, bits_to_symbols, q_from_ber, slice_symbols, statistical_ber,
    symbols_to_bits,
)
from pam4link.prbs import prbs


@pytest.mark.parametrize("order", [7, 9, 11, 13])
def test_prbs_is_maximal_length(order):
    period = 2**order - 1
    seq = prbs(order, 2 * period)
    assert np.array_equal(seq[:period], seq[period:])
    # no shorter period: a maximal-length sequence has 2^(n-1) ones per period
    assert seq[:period].sum() == 2 ** (order - 1)
    for d in (1, 2, 3, period // 3):
        assert not np.array_equal(seq[:period], np.roll(seq[:period], d))


def test_gray_mapping_round_trip_and_adjacency():
    bits = prbs(15, 4000)
    assert np.array_equal(symbols_to_bits(bits_to_symbols(bits)), bits)
    # neighbouring PAM4 levels differ in exactly one bit
    for i in range(3):
        assert np.sum(GRAY_BITS[i] != GRAY_BITS[i + 1]) == 1


def test_slicer_decides_nearest_level():
    y = np.array([-3.4, -2.1, -1.9, -0.1, 0.1, 1.9, 2.1, 3.5])
    assert slice_symbols(y).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert np.array_equal(slice_symbols(LEVELS), np.arange(4))


def test_q_inverse():
    for ber in (1e-2, 2.4e-4, 1e-6, 1e-12):
        assert ber_from_q(q_from_ber(ber)) == pytest.approx(ber, rel=1e-6)


def test_statistical_ber_matches_closed_form_for_ideal_levels():
    rng = np.random.default_rng(0)
    sym = rng.integers(0, 4, 10_000)
    sigma = 0.35
    # outer levels have one neighbour, inner two: average = 1.5 Q / 2 bits
    assert statistical_ber(LEVELS[sym], sym, sigma) == pytest.approx(ber_from_q(1 / sigma), rel=0.05)


def test_butterworth_3db_point():
    fc = 30e9
    h = butterworth(np.array([0.0, fc]), fc, order=2)
    assert abs(h[0]) == pytest.approx(1.0)
    assert 20 * np.log10(abs(h[1])) == pytest.approx(-3.01, abs=0.05)


def test_ctle_peaking_and_unity_dc():
    f = np.linspace(0, 100e9, 20001)
    h = ctle(f, peaking_db=6.0, f_pole1=26.5e9, f_pole2=53e9)
    assert abs(h[0]) == pytest.approx(1.0)
    peak_db = 20 * np.log10(np.abs(h).max())
    assert 2.0 < peak_db < 6.0  # finite poles eat part of the ideal 6 dB


def test_channel_loss_at_nyquist():
    fn = 26.5625e9
    h = channel_loss(np.array([0.0, fn]), 12.0, fn)
    assert 20 * np.log10(abs(h[0])) == pytest.approx(0.0)
    assert 20 * np.log10(abs(h[1])) == pytest.approx(-12.0)


def test_apply_response_identity():
    x = np.random.default_rng(1).standard_normal(1024)
    f = np.fft.rfftfreq(len(x))
    assert np.allclose(apply_response(x, np.ones_like(f)), x)


def test_lms_recovers_known_isi_channel():
    rng = np.random.default_rng(2)
    d = LEVELS[rng.integers(0, 4, 20_000)]
    # channel with one pre-cursor and two post-cursors
    s = 0.15 * np.roll(d, -1) + 1.0 * d + 0.35 * np.roll(d, 1) + 0.1 * np.roll(d, 2)
    taps, sq_err = lms_train(s, d, n_taps=9, n_pre=3, mu=0.02, n_train=10_000)
    assert sq_err[-1000:].mean() < 0.05 * sq_err[:200].mean()
    y = apply_ffe(s, taps, 3)
    assert np.array_equal(slice_symbols(y), slice_symbols(d))

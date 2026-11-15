"""System-level regression: the simulator against theory and against itself."""
from dataclasses import replace

import numpy as np
import pytest

from pam4link import KP4_BER_THRESHOLD, LinkConfig, bathtub, simulate
from pam4link.pam4 import ber_from_q, q_from_ber

# Distortion-free link: only Gaussian receiver noise limits BER.
IDEAL = LinkConfig(
    sps=8, n_symbols=2**16, host_channel_loss_db=0.0, driver_bw_hz=None, driver_compression=0.0,
    tia_bw_hz=None, tia_noise_density=18e-12, ctle_peaking_db=0.0, rj_rms_ui=0.0, sj_pp_ui=0.0, ffe_taps=0,
)

FAST = LinkConfig(sps=16, n_symbols=2**15)


def q_db(ber_a: float, ber_b: float) -> float:
    """Difference between two BERs expressed as an SNR (Q-factor) difference in dB."""
    return float(20 * np.log10(q_from_ber(ber_a) / q_from_ber(ber_b)))


@pytest.mark.parametrize("fiber_loss_db", [8.5, 9.0, 9.5, 10.0])
@pytest.mark.parametrize("seed", [1, 2])
def test_ideal_link_ber_matches_gaussian_theory(fiber_loss_db, seed):
    res = simulate(replace(IDEAL, fiber_loss_db=fiber_loss_db, seed=seed))
    e = res.eval_slice
    y, sym = res.y_clean[e], res.tx_symbols[e]
    half_spacing = (y[sym == 1].mean() - y[sym == 0].mean()) / 2
    theory = ber_from_q(half_spacing / res.sigma)
    assert res.bit_errors >= 30, "operating point gives too few errors to compare"
    assert abs(q_db(res.ber, theory)) < 0.5


def test_ideal_link_without_noise_is_error_free_and_fully_open():
    res = simulate(replace(IDEAL, tia_noise_density=0.0))
    assert res.bit_errors == 0
    assert res.eye_opening() == pytest.approx(1.0, abs=1e-4)


def test_statistical_ber_matches_monte_carlo_on_impaired_link():
    for loss in np.arange(6.0, 14.0, 0.5):
        res = simulate(replace(FAST, fiber_loss_db=float(loss)))
        if 2e-3 <= res.ber_statistical <= 3e-2:
            break
    else:
        pytest.fail("no fiber loss gave a BER in the comparison window")
    assert abs(q_db(res.ber, res.ber_statistical)) < 0.5


def test_equalization_improves_lossy_link():
    lossy = replace(FAST, host_channel_loss_db=14.0, tia_noise_density=0.0, rj_rms_ui=0.0, sj_pp_ui=0.0)
    raw = simulate(replace(lossy, ctle_peaking_db=0.0, ffe_taps=0)).eye_opening()
    ctle_only = simulate(replace(lossy, ffe_taps=0)).eye_opening()
    full = simulate(lossy).eye_opening()
    assert ctle_only > raw
    assert full > ctle_only


def test_default_link_passes_kp4_fec_limit():
    assert simulate(FAST).ber_statistical < KP4_BER_THRESHOLD


def test_bathtub_is_lowest_near_the_sampling_point():
    res = simulate(FAST)
    phases = np.linspace(-0.5, 0.5, 21)
    ber = bathtub(res, phases)
    assert ber[10] <= ber[0] and ber[10] <= ber[-1]
    assert ber[0] > 10 * ber[10]


def test_more_received_power_lowers_ber():
    bers = [simulate(replace(FAST, fiber_loss_db=l)).ber_statistical for l in (14.0, 12.0, 10.0)]
    assert bers[0] > bers[1] > bers[2]

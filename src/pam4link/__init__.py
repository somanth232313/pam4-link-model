"""Behavioral model of a 106.25 Gb/s PAM4 electro-optical link."""
from .link import KP4_BER_THRESHOLD, LinkConfig, LinkResult, bathtub, eye_width, sensitivity_dbm, simulate, sweep_rx_power

__all__ = [
    "KP4_BER_THRESHOLD", "LinkConfig", "LinkResult", "bathtub", "eye_width",
    "sensitivity_dbm", "simulate", "sweep_rx_power",
]

"""Run the PAM4 link in three receiver configurations and save plots + metrics.

    python scripts/run_demo.py            # full run
    python scripts/run_demo.py --quick    # fewer symbols, faster
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pam4link import LinkConfig, bathtub, eye_width, sensitivity_dbm, simulate, sweep_rx_power  # noqa: E402
from pam4link.plots import (  # noqa: E402
    equalized_wave, plot_bathtub, plot_ber_vs_power, plot_eye, plot_ffe, plot_responses,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer symbols and sweep points")
    ap.add_argument("--out", type=Path, default=ROOT / "plots")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = LinkConfig(n_symbols=2**14 if args.quick else 2**16, sps=16 if args.quick else 32)
    configs = {
        "no equalization": replace(base, ctle_peaking_db=0.0, ffe_taps=0),
        "CTLE": replace(base, ffe_taps=0),
        "CTLE + FFE": base,
    }
    results = {name: simulate(cfg) for name, cfg in configs.items()}

    # --- metrics -----------------------------------------------------------
    phases = np.linspace(-0.5, 0.5, 41 if args.quick else 81)
    tubs = {name: bathtub(r, phases) for name, r in results.items()}
    losses = np.arange(6.0, 16.1, 2.0 if args.quick else 1.0)
    sweeps = {name: sweep_rx_power(replace(cfg, n_symbols=min(cfg.n_symbols, 2**15)), losses)
              for name, cfg in configs.items()}

    metrics = {"config": {k: v for k, v in asdict(base).items()}, "results": {}}
    print(f"\nLink: {base.baud * 2 / 1e9:.2f} Gb/s PAM4, rx power {base.rx_power_dbm:.1f} dBm avg\n")
    print(f"{'receiver':<18}{'eye open':>10}{'eye width':>11}{'BER (MC)':>12}{'BER (stat)':>12}{'sens @KP4':>12}")
    for name, r in results.items():
        m = {
            "eye_opening_pct": 100 * r.eye_opening(after_ffe=True),
            "eye_width_ui_at_kp4": eye_width(phases, tubs[name]),
            "ber_monte_carlo": r.ber,
            "bit_errors": r.bit_errors,
            "bits": r.n_bits,
            "ber_statistical": r.ber_statistical,
            "sensitivity_dbm_at_kp4": sensitivity_dbm(sweeps[name]["rx_power_dbm"], sweeps[name]["ber_stat"]),
            "ffe_taps": r.ffe_taps.tolist(),
        }
        metrics["results"][name] = m
        sens = m["sensitivity_dbm_at_kp4"]
        sens_txt = f"{sens:>9.2f}dBm" if np.isfinite(sens) else f"{'never':>12}"
        print(f"{name:<18}{m['eye_opening_pct']:>9.1f}%{m['eye_width_ui_at_kp4']:>9.2f}UI"
              f"{m['ber_monte_carlo']:>12.2e}{m['ber_statistical']:>12.2e}{sens_txt}")
    pre = 100 * results["CTLE"].eye_opening(after_ffe=True)
    post = metrics["results"]["CTLE + FFE"]["eye_opening_pct"]
    metrics["ffe_eye_opening_gain_pct_points"] = post - pre
    print(f"\nFFE changes vertical eye opening from {pre:.1f}% (CTLE only) to {post:.1f}% "
          f"({post - pre:+.1f} percentage points)")
    for m in metrics["results"].values():  # NaN is not valid JSON
        if not np.isfinite(m["sensitivity_dbm_at_kp4"]):
            m["sensitivity_dbm_at_kp4"] = None
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- figures -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_eye(axes[0], results["no equalization"], title="Receiver input (no EQ)")
    plot_eye(axes[1], results["CTLE"], title="After CTLE")
    plot_eye(axes[2], results["CTLE + FFE"], wave=equalized_wave(results["CTLE + FFE"]), title="After CTLE + FFE")
    fig.tight_layout()
    fig.savefig(args.out / "eye_diagrams.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_bathtub(ax, phases, tubs)
    fig.tight_layout()
    fig.savefig(args.out / "bathtub.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_ber_vs_power(ax, sweeps)
    fig.tight_layout()
    fig.savefig(args.out / "ber_vs_power.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_responses(ax, results["CTLE + FFE"])
    fig.tight_layout()
    fig.savefig(args.out / "frequency_response.png", dpi=130)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_ffe(a1, a2, results["CTLE + FFE"])
    fig.tight_layout()
    fig.savefig(args.out / "ffe.png", dpi=130)

    print(f"\nSaved plots and metrics.json to {args.out}")


if __name__ == "__main__":
    main()

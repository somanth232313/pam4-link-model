# pam4-link-model

End-to-end Python behavioral model of a **106.25 Gb/s PAM4 electro-optical link**, covering transmitter, channel, receiver, equalization, jitter and BER analysis.

> 🚧 Work in progress.

## Plan
- [ ] PRBS7/PRBS31 generator and PAM4 Gray-coded symbol mapping
- [ ] Transmitter: bandwidth-limited modulator driver with nonlinearity
- [ ] Channel: frequency-dependent loss
- [ ] Receiver: photodiode + TIA noise model
- [ ] Equalization: CTLE and LMS-adapted FFE
- [ ] Clock jitter: random + deterministic
- [ ] Analysis: eye diagrams, bathtub curves, BER vs. received power
- [ ] pytest regression: simulated BER vs. analytical Gaussian-noise theory

## Layout
`src/` link blocks · `tests/` pytest regression · `notebooks/` experiments · `plots/` results
